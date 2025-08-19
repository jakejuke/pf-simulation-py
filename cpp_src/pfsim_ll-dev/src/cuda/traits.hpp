#ifndef SRC_CUDA_TRAITS_HPP
#define SRC_CUDA_TRAITS_HPP

// ###########################
// STARTED WITH THIS FILE AN NEVER FINISHED
// DO NOT INCLUDE ANYWHERE
// ###########################

namespace traits {

// define the Require keyword    
template <typename T, typename... Args>
using Require = typename std::common_type<T, Args...>::type;    

// very basic traits used for
// further step
template <typename T>
struct make_ElementType
{
};

template <template<typename> class X, typename T>
struct make_ElementType<X<T>>
{
    using type = T;
};    

template <typename T>
using ElementType
    = typename make_ElementType<typename std::remove_reference<T>::type>::type;

// "Require" for same element type
template <typename TX, typename TY>
using SameElementType
    =  typename std::conditional<
                std::is_same<ElementType<TX>, ElementType<TY>>::value,
                bool, void>::type;

// define for mesh2d
template <typename T>
using Mesh 
            =
            std::enable_if<std::remove_reference<T>::type::is_Mesh2d::value,
                    bool>;

template <typename T>
using HostMesh = typename traits::Mesh<T>;

template <typename T>
using HostOnlyMesh 
            =
            std::enable_if<std::remove_reference<T>::type::is_HostOnlyMesh2d::value,
                    bool>;

template <typename T>
using DeviceMesh
        =
        std::enable_if<std::remove_reference<T>::type::is_DeviceMesh2d::value,
                    bool>;

}

#endif
