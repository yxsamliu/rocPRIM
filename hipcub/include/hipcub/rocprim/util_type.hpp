// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPCUB_ROCPRIM_UTIL_TYPE_HPP_
#define HIPCUB_ROCPRIM_UTIL_TYPE_HPP_

#include <type_traits>

#include "../config.hpp"

BEGIN_HIPCUB_NAMESPACE

using NullType = ::rocprim::empty_type;

template<bool B, typename T, typename F>
struct If
{
    using Type = typename std::conditional<B, T, F>::type;
};

template<typename T>
struct IsPointer
{
    static constexpr bool VALUE = std::is_pointer<T>::value;
};

template<typename T>
struct IsVolatile
{
    static constexpr bool VALUE = std::is_volatile<T>::value;
};

template<typename T>
struct RemoveQualifiers
{
    using Type = typename std::remove_cv<T>::type;
};

template<int N>
struct PowerOfTwo
{
    static constexpr bool VALUE = ::rocprim::detail::is_power_of_two<N>();
};

namespace detail
{

template<int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2Impl
{
    static constexpr int VALUE = Log2Impl<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE;
};

template<int N, int COUNT>
struct Log2Impl<N, 0, COUNT>
{
    static constexpr int VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1;
};

} // end of detail namespace

template<int N>
struct Log2
{
    static_assert(N != 0, "The logarithm of zero is undefined");
    static constexpr int VALUE = detail::Log2Impl<N>::VALUE;
};

template<typename T>
struct DoubleBuffer
{
    T * d_buffers[2];

    int selector;

    HIPCUB_HOST_DEVICE inline
    DoubleBuffer()
    {
        selector = 0;
        d_buffers[0] = nullptr;
        d_buffers[1] = nullptr;
    }

    HIPCUB_HOST_DEVICE inline
    DoubleBuffer(T * d_current, T * d_alternate)
    {
        selector = 0;
        d_buffers[0] = d_current;
        d_buffers[1] = d_alternate;
    }

    HIPCUB_HOST_DEVICE inline
    T * Current()
    {
        return d_buffers[selector];
    }

    HIPCUB_HOST_DEVICE inline
    T * Alternate()
    {
        return d_buffers[selector ^ 1];
    }
};

template<
    class Key,
    class Value
>
using KeyValuePair = ::rocprim::key_value_pair<Key, Value>;

namespace detail
{

template<typename T>
inline
::rocprim::double_buffer<T> to_double_buffer(DoubleBuffer<T>& source)
{
    return ::rocprim::double_buffer<T>(source.Current(), source.Alternate());
}

template<typename T>
inline
void update_double_buffer(DoubleBuffer<T>& target, ::rocprim::double_buffer<T>& source)
{
    if(target.Current() != source.current())
    {
        target.selector ^= 1;
    }
}

}

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_UTIL_TYPE_HPP_
