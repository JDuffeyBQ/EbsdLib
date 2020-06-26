#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "EbsdLib/Core/Orientation.hpp"
#include "EbsdLib/Core/OrientationTransformation.hpp"
#include "EbsdLib/Core/Quaternion.hpp"

namespace py = pybind11;

template <class Container>
py::array_t<typename Container::value_type> make_pyarray(Container&& vec)
{
  Container* vecPtr = new Container(std::move(vec));
  py::capsule capsule(vecPtr, [](void* p) { delete reinterpret_cast<Container*>(p); });
  return py::array_t<typename Container::value_type>(vecPtr->size(), vecPtr->data(), capsule);
}

template <class Container>
py::array_t<typename Container::value_type> make_pyarray(Container&& vec, const py::array::ShapeContainer& shape)
{
  Container* vecPtr = new Container(std::move(vec));
  py::capsule capsule(vecPtr, [](void* p) { delete reinterpret_cast<Container*>(p); });
  return py::array_t<typename Container::value_type>(shape, vecPtr->data(), capsule);
}

enum class Rotation : int32_t
{
  Active = -1,
  Passive = 1
};

enum class QuatOrder : QuatF::EnumType
{
  ScalarVector = static_cast<QuatF::EnumType>(QuatF::Order::ScalarVector),
  VectorScalar = static_cast<QuatF::EnumType>(QuatF::Order::VectorScalar)
};

template <class T>
void bindQuaternion(py::module& m, const char* name)
{
  using Quat = Quaternion<T>;
  py::class_<Quat>(m, name)
      .def(py::init<>())
      .def(py::init<T, T, T, T>())
      .def(py::init([](py::list list, QuatOrder order) {
        if(list.size() != 4)
        {
          throw std::runtime_error("Size must be 4");
        }

        Quat quat;

        switch(order)
        {
        case QuatOrder::ScalarVector:
          quat.x() = py::cast<T>(list[1]);
          quat.y() = py::cast<T>(list[2]);
          quat.z() = py::cast<T>(list[3]);
          quat.w() = py::cast<T>(list[0]);
          break;
        case QuatOrder::VectorScalar:
          quat.x() = py::cast<T>(list[0]);
          quat.y() = py::cast<T>(list[1]);
          quat.z() = py::cast<T>(list[2]);
          quat.w() = py::cast<T>(list[3]);
          break;
        }
        return quat;
      }))
      .def(py::init([](py::array_t<T> data, QuatOrder order) {
        if(data.ndim() != 1)
        {
          throw std::runtime_error("Number of dimensions must be one");
        }
        
        if(data.size() != 4)
        {
          throw std::runtime_error("Size must be 4");
        }

        return Quat(data.data(), static_cast<Quat::Order>(order));
      }))
      .def_property(
          "x", [](const Quat& self) { return self.x(); }, [](Quat& self, T value) { self.x() = value; })
      .def_property(
          "y", [](const Quat& self) { return self.y(); }, [](Quat& self, T value) { self.y() = value; })
      .def_property(
          "z", [](const Quat& self) { return self.z(); }, [](Quat& self, T value) { self.z() = value; })
      .def_property(
          "w", [](const Quat& self) { return self.w(); }, [](Quat& self, T value) { self.w() = value; })
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self - py::self)
      .def(py::self -= py::self)
      .def(py::self * py::self)
      .def(py::self *= py::self)
      .def(-py::self)
      .def_static("identity", &Quat::identity)
      .def("elementWiseAbs", &Quat::elementWiseAbs)
      .def("scalarMultiply", &Quat::scalarMultiply)
      .def("scalarDivide", &Quat::scalarDivide)
      .def("scalarAdd", &Quat::scalarAdd)
      .def("elementWiseAssign", &Quat::elementWiseAssign)
      .def("negate", &Quat::negate)
      .def("conjugate", &Quat::conjugate)
      .def("norm", &Quat::norm)
      .def("length", &Quat::length)
      .def("unitQuaternion", &Quat::unitQuaternion)
      .def("getMisorientationVector", &Quat::getMisorientationVector)
      .def("multiplyByVector", [](Quat& self, const std::array<T, 3>& vec) { return self.multiplyByVector(vec.data()); })
      .def("rotateVector", [](Quat& self, const std::array<T, 3>& vec, Rotation rot) { return self.rotateVector(vec.data(), static_cast<int32_t>(rot)); })
      .def("to_array",
           [](const Quat& self, QuatOrder order) {
             py::array_t<T> data(4);
             T* ptr = data.mutable_data();
             self.copyInto(ptr, static_cast<Quat::Order>(order));
             return data;
           })
      .def("__repr__", [](const Quat& self) {
        std::stringstream ss;
        ss << "(<" << self.x() << ", " << self.y() << ", " << self.z() << ">, " << self.w() << ")";
        return ss.str();
      });
}

template <class T>
void bind_eu2qu(py::module& m)
{
  m.def("eu2qu", [](py::array_t<T, py::array::c_style | py::array::forcecast> data) {
    if(data.ndim() != 1)
    {
      throw std::runtime_error("Number of dimensions must be one");
    }

    if(data.size() != 3)
    {
      throw std::runtime_error("Size must be 3");
    }

    Orientation<T> orientation(data.mutable_data(), static_cast<size_t>(data.size()));
    return OrientationTransformation::eu2qu<Orientation<T>, Quaternion<T>>(orientation, Quaternion<T>::Order::VectorScalar);
  });
}

template <class T>
void bind_eu2om(py::module& m)
{
  m.def("eu2om", [](py::array_t<T, py::array::c_style | py::array::forcecast> data) {
    if(data.ndim() != 1)
    {
      throw std::runtime_error("Number of dimensions must be one");
    }

    if(data.size() != 3)
    {
      throw std::runtime_error("Size must be 3");
    }

    Orientation<T> orientation(data.mutable_data(), static_cast<size_t>(data.size()));

    auto output = OrientationTransformation::eu2om<Orientation<T>, Orientation<T>>(orientation);

    return make_pyarray(std::move(output), {3, 3});
  });
}

template <class T>
void bind_eu2ax(py::module& m)
{
  m.def("eu2ax", [](py::array_t<T, py::array::c_style | py::array::forcecast> data) {
    if(data.ndim() != 1)
    {
      throw std::runtime_error("Number of dimensions must be one");
    }

    if(data.size() != 3)
    {
      throw std::runtime_error("Size must be 3");
    }

    Orientation<T> orientation(data.mutable_data(), static_cast<size_t>(data.size()));

    auto output = OrientationTransformation::eu2ax<Orientation<T>, Orientation<T>>(orientation);

    return make_pyarray(std::move(output));
  });
}

template <class T>
void bind_eu2ro(py::module& m)
{
  m.def("eu2ro", [](py::array_t<T, py::array::c_style | py::array::forcecast> data) {
    if(data.ndim() != 1)
    {
      throw std::runtime_error("Number of dimensions must be one");
    }

    if(data.size() != 3)
    {
      throw std::runtime_error("Size must be 3");
    }

    Orientation<T> orientation(data.mutable_data(), static_cast<size_t>(data.size()));

    auto output = OrientationTransformation::eu2ro<Orientation<T>, Orientation<T>>(orientation);

    return make_pyarray(std::move(output));
  });
}

template <class T>
void bind_eu2ho(py::module& m)
{
  m.def("eu2ho", [](py::array_t<T, py::array::c_style | py::array::forcecast> data) {
    if(data.ndim() != 1)
    {
      throw std::runtime_error("Number of dimensions must be one");
    }

    if(data.size() != 3)
    {
      throw std::runtime_error("Size must be 3");
    }

    Orientation<T> orientation(data.mutable_data(), static_cast<size_t>(data.size()));

    auto output = OrientationTransformation::eu2ho<Orientation<T>, Orientation<T>>(orientation);

    return make_pyarray(std::move(output));
  });
}

template <class T>
void bind_eu2cu(py::module& m)
{
  m.def("eu2cu", [](py::array_t<T, py::array::c_style | py::array::forcecast> data) {
    if(data.ndim() != 1)
    {
      throw std::runtime_error("Number of dimensions must be one");
    }

    if(data.size() != 3)
    {
      throw std::runtime_error("Size must be 3");
    }

    Orientation<T> orientation(data.mutable_data(), static_cast<size_t>(data.size()));

    auto output = OrientationTransformation::eu2cu<Orientation<T>, Orientation<T>>(orientation);

    return make_pyarray(std::move(output));
  });
}

PYBIND11_MODULE(ebsdlib, m)
{
  py::enum_<Rotation>(m, "Rotation").value("Active", Rotation::Active).value("Passive", Rotation::Passive);
  py::enum_<QuatOrder>(m, "QuatOrder").value("ScalarVector", QuatOrder::ScalarVector).value("VectorScalar", QuatOrder::VectorScalar);

  bindQuaternion<float>(m, "QuatF");
  bindQuaternion<double>(m, "QuatD");

  bind_eu2qu<float>(m);
  bind_eu2qu<double>(m);

  bind_eu2om<float>(m);
  bind_eu2om<double>(m);

  bind_eu2ax<float>(m);
  bind_eu2ax<double>(m);

  bind_eu2ro<float>(m);
  bind_eu2ro<double>(m);

  bind_eu2ho<float>(m);
  bind_eu2ho<double>(m);

  bind_eu2cu<float>(m);
  bind_eu2cu<double>(m);
}
