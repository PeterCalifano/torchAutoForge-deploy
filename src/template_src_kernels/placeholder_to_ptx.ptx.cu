/**
 * @file placeholder_to_ptx.ptx.cu
 * @brief Retained compile-and-embed PTX build example.
 */

namespace placeholder_cuda
{
    /**
     * @brief Minimal kernel compiled to PTX and embedded in the library.
     *
     * The example intentionally has no runtime loader. It demonstrates the
     * dedicated `.ptx.cu` build path independently of any renderer API.
     */
    extern "C" __global__ void ptafdeploy_ptx_example_kernel()
    {
    }
} // namespace placeholder_cuda
