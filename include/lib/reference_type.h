#include <functional>

template <typename T> class RefT : public std::reference_wrapper<T> {
public:
  RefT(T &v) : std::reference_wrapper<T>(v) {}
  T *operator->() { return &this->get(); }
  const T *operator->() const { return &this->get(); }
};
