/*! \file name_value_pair.hpp
    \brief NameValuePair class and helpers */
/*
  Copyright (c) 2014, Randolph Voorhies, Shane Grant
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
      * Neither the name of cereal nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL RANDOLPH VOORHIES OR SHANE GRANT BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef CEREAL_NAME_VALUE_PAIR_HPP_
#define CEREAL_NAME_VALUE_PAIR_HPP_

#include <type_traits>
#include <utility>

namespace cereal {
  // ######################################################################
  namespace detail
  {
    struct NameValuePairCore {}; //!< Traits struct for NVPs
    struct DeferredDataCore {}; //!< Traits struct for DeferredData
  }

  // ######################################################################
  //! For holding name value pairs
  /*! This pairs a name (some string) with some value such that an archive
      can potentially take advantage of the pairing.

      In serialization functions, NameValuePairs are usually created like so:
      @code{.cpp}
      struct MyStruct
      {
        int a, b, c, d, e;

        template<class Archive>
        void serialize(Archive & archive)
        {
          archive( CEREAL_NVP(a),
                   CEREAL_NVP(b),
                   CEREAL_NVP(c),
                   CEREAL_NVP(d),
                   CEREAL_NVP(e) );
        }
      };
      @endcode

      Alternatively, you can give you data members custom names like so:
      @code{.cpp}
      struct MyStruct
      {
        int a, b, my_embarrassing_variable_name, d, e;

        template<class Archive>
        void serialize(Archive & archive)
        {
          archive( CEREAL_NVP(a),
                   CEREAL_NVP(b),
                   cereal::make_nvp("var", my_embarrassing_variable_name) );
                   CEREAL_NVP(d),
                   CEREAL_NVP(e) );
        }
      };
      @endcode

      There is a slight amount of overhead to creating NameValuePairs, so there
      is a third method which will elide the names when they are not used by
      the Archive:

      @code{.cpp}
      struct MyStruct
      {
        int a, b;

        template<class Archive>
        void serialize(Archive & archive)
        {
          archive( cereal::make_nvp<Archive>(a),
                   cereal::make_nvp<Archive>(b) );
        }
      };
      @endcode

      This third method is generally only used when providing generic type
      support.  Users writing their own serialize functions will normally
      explicitly control whether they want to use NVPs or not.

      @internal */
  template <class T>
  class NameValuePair : detail::NameValuePairCore
  {
    private:
      // If we get passed an array, keep the type as is, otherwise store
      // a reference if we were passed an l value reference, else copy the value
      using Type = typename std::conditional<std::is_array<typename std::remove_reference<T>::type>::value,
                                             typename std::remove_cv<T>::type,
                                             typename std::conditional<std::is_lvalue_reference<T>::value,
                                                                       T,
                                                                       typename std::decay<T>::type>::type>::type;

      // prevent nested nvps
      static_assert( !std::is_base_of<detail::NameValuePairCore, T>::value,
                     "Cannot pair a name to a NameValuePair" );

      NameValuePair & operator=( NameValuePair const & ) = delete;

    public:
      //! Constructs a new NameValuePair
      /*! @param n The name of the pair
          @param v The value to pair.  Ideally this should be an l-value reference so that
                   the value can be both loaded and saved to.  If you pass an r-value reference,
                   the NameValuePair will store a copy of it instead of a reference.  Thus you should
                   only pass r-values in cases where this makes sense, such as the result of some
                   size() call.
          @internal */
      NameValuePair( char const * n, T && v ) : name(n), value(std::forward<T>(v)) {}

      char const * name;
      Type value;
  };

  //! Creates a name value pair
  /*! @relates NameValuePair
      @ingroup Utility */
  template <class T> inline
  NameValuePair<T> make_nvp( const char * name, T && value )
  {
    return {name, std::forward<T>(value)};
  }

  //! Creates a name value pair for the variable T with the same name as the variable
  /*! @relates NameValuePair
      @ingroup Utility */
  #define CEREAL_NVP(T) ::cereal::make_nvp(#T, T)
}
#endif // CEREAL_NAME_VALUE_PAIR_HPP_
