#ifndef SRC_AUX_SLICE_HPP
#define SRC_AUX_SLICE_HPP 1

#include <cassert>
#include <cstdlib>
#include <type_traits>

/*
   Credits: A.F.Borchert, provided in the
   course of High Performance Computing I at
   Ulm University in winter term 2020/2021;
   
   UniformSlices and AlignedSlices support partitioning where
    - N, i.e. the problem size (e.g. a dimension of a matrix or vector), and
    - P, i.e. the number of partitions
   are fixed. Both classes deliver then
    - Oi, i.e. the offset for the i-th partition, and
    - Li, i.e. the length of the i-th partition
   for i = 0, ..., P-1 where
    - sum Li = P
    - Oi = sum Lj for j = 0, ..., i-1, and
    - Li >= Lj if i <= j
*/

namespace aux {

template<typename T>
T align(T size, T alignment) {
   T remainder = size % alignment;
   if (remainder == 0) {
      return size;
   } else {
      return size + alignment - remainder;
   }
}

/* common base class of UniformSlices & AlignedSlices */
template<typename T = std::size_t>
struct BasicSlices {
   using Index = T;
   BasicSlices(T nof_partitions, T problem_size, T granularity = 1) :
	 nof_partitions((assert(nof_partitions > 0), nof_partitions)),
	 problem_size(problem_size),
	 granularity(granularity),
	 slice_size{
	    problem_size / granularity >= nof_partitions?
	       align(T((problem_size + nof_partitions - 1) /
		  nof_partitions), granularity)
	       :
		  granularity} {
   }
   const T nof_partitions; const T problem_size; const T granularity;
   const T slice_size;
};

template<typename T = std::size_t>
class UniformSlices : public BasicSlices<T> {
   using BasicSlices<T>::nof_partitions;
   using BasicSlices<T>::problem_size;
   using BasicSlices<T>::slice_size;
   using BasicSlices<T>::granularity;

   public:
      UniformSlices(T nof_partitions, T problem_size, T granularity = 1) :
	    BasicSlices<T>(nof_partitions, problem_size, granularity) {
	 /*
	    full_slices: number of slices with a size of slice_size
	    regular_slices: number of slices with a size of slice_size
			    or slice_size - granularity
	 */
	 if (nof_partitions == 1 || granularity > problem_size) {
	    // special case where we do not partitionate
	    full_slices = 0; regular_slices = 0;
	 } else {
	    T remainder = problem_size % granularity;
	    if (remainder == 0 &&
		  problem_size / granularity % nof_partitions == 0) {
	       // trivial case
	       full_slices = regular_slices = nof_partitions;
	    } else {
	       full_slices = problem_size / granularity % nof_partitions;
	       if (slice_size == granularity || slice_size > problem_size) {
		  regular_slices = full_slices;
	       } else {
		  T smaller_regular_slices =
		     (problem_size - slice_size * full_slices) /
			(slice_size - granularity);
		  if (remainder && smaller_regular_slices > 0) {
		     --smaller_regular_slices;
		     if (full_slices + 1 < nof_partitions) {
			++full_slices;
			if (smaller_regular_slices > 0) {
			   --smaller_regular_slices;
			}
		     }
		  }
		  regular_slices = full_slices + smaller_regular_slices;
	       }
	    }
	 }
      }
      T offset(T index) const {
	 assert(index < nof_partitions);
	 if (index < full_slices) {
	    return index * slice_size;
	 } else if (index <= regular_slices) {
	    return full_slices * slice_size +
		   (index - full_slices) * (slice_size - granularity);
	 } else {
	    return problem_size;
	 }
      }
      T size(T index) const {
	 assert(index < nof_partitions);
	 if (index < full_slices) {
	    return slice_size;
	 } else if (index < regular_slices) {
	    return slice_size - granularity;
	 } else if (index == regular_slices) {
	    return problem_size - offset(index);
	 } else {
	    return 0;
	 }
      }

   private:
      T full_slices;
      T regular_slices;
};

template<typename T = std::size_t>
struct AlignedSlices : public BasicSlices<T> {
   using BasicSlices<T>::nof_partitions;
   using BasicSlices<T>::problem_size;
   using BasicSlices<T>::slice_size;

   AlignedSlices(T nof_partitions, T problem_size, T granularity = 1) :
	 BasicSlices<T>(nof_partitions, problem_size, granularity),
	 full_slices(problem_size / slice_size) {
   }
   T offset(T index) const {
      assert(index < nof_partitions);
      if (slice_size > 0) {
	 if (index <= full_slices) {
	    return index * slice_size;
	 } else {
	    return problem_size;
	 }
      } else {
	 return index;
      }
   }
   T size(T index) const {
      assert(index < nof_partitions);
      if (index < full_slices) {
	 return slice_size;
      } else if (index == full_slices) {
	 return problem_size - full_slices * slice_size;
      } else {
	 return 0;
      }
   }
   const T full_slices;
};

template<typename Slices, typename Body>
void foreach_slice(typename Slices::Index nof_partitions,
      typename Slices::Index problem_size, Body body) {
   Slices slices{nof_partitions, problem_size};
   for (typename Slices::Index index = 0; index < nof_partitions; ++index) {
      auto size = slices.size(index);
      if (size > 0) {
	 body(slices.offset(index), size);
      }
   }
}
template<template<typename> class Slices, typename Body>
void foreach_slice(std::size_t nof_partitions, std::size_t problem_size,
      Body body) {
   foreach_slice<Slices<std::size_t>>(nof_partitions, problem_size, body);
}

template<typename Slices, typename Body>
void foreach_slice(typename Slices::Index nof_partitions,
      typename Slices::Index problem_size, typename Slices::Index granularity,
      Body body) {
   Slices slices{nof_partitions, problem_size, granularity};
   for (typename Slices::Index index = 0; index < nof_partitions; ++index) {
      auto size = slices.size(index);
      if (size > 0) {
	 body(slices.offset(index), size);
      }
   }
}

template<template<typename> class Slices, typename Body>
void foreach_slice(std::size_t nof_partitions,
      std::size_t problem_size, std::size_t granularity,
      Body body) {
   foreach_slice<Slices<std::size_t>>(nof_partitions, problem_size,
      granularity, body);
}

} /* namespace aux*/

#endif // SRC_AUX_SLICE_HPP
