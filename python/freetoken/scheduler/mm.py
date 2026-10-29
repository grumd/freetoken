"""Image inputs of a request: per-image content digests, the prefix-cache key they induce,
and the soft-token embeddings, produced only for images the forward actually runs.

An image whose placeholders all sit inside the request's prefix-cache hit is never encoded;
the rest go through the vision tower at admission (see ``Scheduler._embed_images``)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch


@dataclass(eq=False)
class MMImage:
    digest: bytes
    first: int  # ordinal of this image's first placeholder among the prompt's placeholders
    count: int
    # this image's patch rows and grid until admission; None after it, and when the
    # embeddings came precomputed (offline)
    pixel_values: torch.Tensor | None = None
    grid_thw: torch.Tensor | None = None
    # [count, hidden] on the engine device while some placeholder is still to be forwarded
    embeds: torch.Tensor | None = None


@dataclass(eq=False)
class MMInputs:
    images: List[MMImage]
    positions: torch.Tensor  # prompt positions of the placeholders, ascending (CPU int64)
    key: torch.Tensor  # prefix-cache key, see mm_cache_key

    def span(self, img: MMImage, start: int, end: int) -> Tuple[int, int]:
        """Rows [lo, hi) of ``img`` whose placeholders fall in prompt positions [start, end)."""
        pos = self.positions[img.first : img.first + img.count]
        lo = int(torch.searchsorted(pos, start))
        hi = int(torch.searchsorted(pos, end))
        return lo, hi

    def missing(self, cached_len: int) -> List[MMImage]:
        """Images with a placeholder at or past ``cached_len`` and no embeddings yet."""
        return [
            img for img in self.images
            if img.embeds is None and int(self.positions[img.first + img.count - 1]) >= cached_len
        ]


def image_digests(
    pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, num_image_tokens: int
) -> List[Tuple[bytes, int, int]] | None:
    """Per image: (content digest, first patch row, soft-token count). None when the patch rows
    do not split into a whole number of rows per soft token, so per-image spans are unknown."""
    rows = image_grid_thw.prod(dim=-1).tolist()
    total = sum(rows)
    if num_image_tokens <= 0 or total != pixel_values.shape[0] or total % num_image_tokens:
        return None
    per_token = total // num_image_tokens
    if any(r % per_token for r in rows):
        return None
    out, start = [], 0
    for r, thw in zip(rows, image_grid_thw):
        h = hashlib.blake2b(digest_size=32)
        h.update(str(pixel_values.dtype).encode())
        h.update(thw.to(torch.int64).numpy().tobytes())
        h.update(pixel_values[start : start + r].contiguous().view(-1).view(torch.uint8).numpy())
        out.append((h.digest(), start, r // per_token))
        start += r
    return out


def nonce_digest(nonce: int) -> bytes:
    return hashlib.blake2b(b"freetoken-mm-nonce" + nonce.to_bytes(8, "little"), digest_size=32).digest()


def mm_cache_key(
    input_ids: torch.Tensor, positions: torch.Tensor, images: Sequence[Tuple[bytes, int]]
) -> torch.Tensor:
    """``input_ids`` with each image's placeholder tokens replaced by values drawn from that
    image's digest. Placeholders share one token id whatever the image, so the radix trees
    must key on content instead; the values are negative so no real token can match them."""
    key = input_ids.clone()
    start = 0
    for digest, n in images:
        words = torch.frombuffer(bytearray(digest), dtype=torch.int32).to(torch.int64)
        vals = -(words & 0x7FFFFFFF) - 1
        reps = (n + vals.numel() - 1) // vals.numel()
        key[positions[start : start + n]] = vals.repeat(reps)[:n].to(key.dtype)
        start += n
    return key

