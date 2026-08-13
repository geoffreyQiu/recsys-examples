# NVE's default feature set builds optional parameter-server plugins. The HSTU
# standalone AOTI path uses LinearUVM layers only, so keep the default key/cache
# kernels while excluding those unused plugin dependencies.
set(
  NVE_FEATURES
  "ht_mask_64;ht_key_64;ht_part_fnv1a;ht_kernel_64;ht_key_32;ht_kernel_128"
  CACHE STRING "NVE features required by standalone HSTU AOTI" FORCE
)
