# RapidJSON provenance

Upstream: Tencent/rapidjson. Headers and license copied from the local
`dev-tools/rapidjson` checkout at revision
`24b5e7a8b27f42fa16b96fc70aade9106cf7102f`.

The header files match the local
`spectral_raytracer_factorized_radiometry_develop/lib/header_only/rapidjson`
distribution byte for byte. The recorded raytracer import revision is
`b50e068e81d32e7f79bb89089305200b2ec892d1`.
Only headers and the matching license are included. This dependency is private
to the optional inference_output implementation; it is not part of any public header.

The local future-onboard-sw donor also uses RapidJSON in its navigation-filter
configuration tests (`codegen/nav_filter/tests`). No code from those tests or
from donor build configuration was imported.
