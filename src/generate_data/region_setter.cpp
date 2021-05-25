#include "generate_data/region_setter.h"

#include "generate_data/constants.h"
#include "generate_data/get_points_from_subset.h"
#include "generate_data/value_adder.h"
#include "lib/projection.h"
#include "meta/array_cat.h"

#include <ATen/ATen.h>
#include <Eigen/Dense>
#include <boost/geometry.hpp>
#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/unpack.hpp>

#include <cmath>

namespace generate_data {
template <size_t n>
inline std::array<at::indexing::TensorIndex, n>
as_tensor_index(std::array<TorchIdxT, n> idxs) {
  return boost::hana::unpack(
      idxs, [&](auto... idxs) -> std::array<at::indexing::TensorIndex, n> {
        return {idxs...};
      });
}

template <unsigned n_prior_dims>
RegionSetter<n_prior_dims>::RegionSetter(
    const std::array<TorchIdxT, n_prior_dims> &prior_dims) {
  unsigned running = 1;
  for (unsigned i = prior_dims.size() - 1;
       i != std::numeric_limits<unsigned>::max(); --i) {
    index_multipliers[i] = running;
    running *= unsigned(prior_dims[i]);
  }
  unsigned total_size = running;
  point_values.resize(total_size);

  std::array<TorchIdxT, n_prior_dims + 1> feature_dims;
  std::copy(prior_dims.begin(), prior_dims.end(), feature_dims.begin());
  feature_dims[n_prior_dims] = constants.n_poly_feature_values;
  overall_features = at::empty(feature_dims);
  counts = at::empty(prior_dims, long_tensor_type);
}

template <unsigned n_prior_dims>
double RegionSetter<n_prior_dims>::set_region(
    const std::array<TorchIdxT, n_prior_dims> &prior_idxs_in,
    const TriangleSubset &region, const Triangle &tri) {
  return region.visit_tagged([&](auto tag, const auto &value) {
    auto prior_idxs = as_tensor_index(prior_idxs_in);

    unsigned total_idx = 0;
    for (unsigned i = 0; i < n_prior_dims; ++i) {
      total_idx += index_multipliers[i] * prior_idxs_in[i];
    }
    auto &item = point_values[total_idx];

    double area_3d = 0.;
    if constexpr (tag == TriangleSubsetType::None) {
      counts.index_put_(prior_idxs, 0);
      item.clear();
    } else {
      const auto pb = get_points_from_subset_with_baryo(tri, region);

      auto feature_idxs =
          as_tensor_index(array_cat(prior_idxs_in, std::array{TorchIdxT(0)}));

      auto feature_adder = make_value_adder([&](float v, int idx) {
        feature_idxs[n_prior_dims] = idx;
        overall_features.index_put_(feature_idxs, v);
      });

      Eigen::Vector3d centroid3d = Eigen::Vector3d::Zero();
      Eigen::Vector2d centroid2d = Eigen::Vector2d::Zero();
      for (const auto &p : pb.points) {
        centroid3d += p;
      }
      for (const auto &p : pb.baryo) {
        centroid2d += Eigen::Vector2d{p.x(), p.y()};
      }
      centroid3d /= pb.points.size();
      centroid2d /= pb.points.size();

      feature_adder.add_values(centroid3d);
      feature_adder.add_values(centroid2d);

      counts.index_put_(prior_idxs, TorchIdxT(pb.baryo.size()));

      // get areas in baryo and in 3d
      const TriPolygon *baryo_poly;
      if constexpr (tag == TriangleSubsetType::All) {
        baryo_poly = &full_triangle;
      } else {
        static_assert(tag == TriangleSubsetType::Some);

        baryo_poly = &value;
      }
      auto rot = find_rotate_vector_to_vector(
          tri.normal(), UnitVectorGen<double>::new_normalize({0., 0., 1.}));
      Eigen::Vector2d vec0 =
          (rot * (tri.vertices[1] - tri.vertices[0])).head(2);
      Eigen::Vector2d vec1 =
          (rot * (tri.vertices[2] - tri.vertices[0])).head(2);
      TriPolygon poly_distorted;
      auto add_point_to_distorted = [&](const BaryoPoint &p) {
        Eigen::Vector2d new_p = p.x() * vec0 + p.y() * vec1;
        boost::geometry::append(poly_distorted,
                                BaryoPoint{new_p.x(), new_p.y()});
      };
      for (const auto &p : pb.baryo) {
        add_point_to_distorted(p);
      }
      add_point_to_distorted(pb.baryo[0]);

      feature_adder.add_value(boost::geometry::area(*baryo_poly));
      area_3d = boost::geometry::area(poly_distorted);
      feature_adder.add_value(area_3d);

      item.resize(pb.baryo.size() * constants.n_poly_point_values);

      auto value_adder =
          make_value_adder([&](float v, int idx) { item[idx] = v; });
      for (unsigned i = 0; i < pb.baryo.size(); ++i) {
        unsigned i_prev = (i == 0) ? pb.baryo.size() - 1 : i - 1;
        unsigned i_next = (i + 1) % pb.baryo.size();

        const auto baryo_prev = baryo_to_eigen(pb.baryo[i_prev]);
        const auto baryo = baryo_to_eigen(pb.baryo[i]);
        const auto baryo_next = baryo_to_eigen(pb.baryo[i_next]);

        Eigen::Vector2d edge_l = baryo_prev - baryo;
        double norm_l = edge_l.norm();
        value_adder.add_value(norm_l);
        Eigen::Vector2d normalized_l = edge_l.normalized();
        value_adder.add_values(normalized_l);

        Eigen::Vector2d edge_r = baryo_next - baryo;
        double norm_r = edge_r.norm();
        value_adder.add_value(norm_r);
        Eigen::Vector2d normalized_r = edge_r.normalized();
        value_adder.add_values(edge_r);

        double dotted = normalized_l.dot(normalized_r);

        value_adder.add_value(dotted);
        value_adder.add_value(std::acos(dotted));
        value_adder.add_values(baryo);
        value_adder.add_values(pb.points[i]);
      }
      debug_assert(value_adder.idx == int(item.size()));
      debug_assert(feature_adder.idx == constants.n_poly_feature_values);
    }

    return area_3d;
  });
}

template <unsigned n_prior_dims>
ATTR_NO_DISCARD_PURE PolygonInput RegionSetter<n_prior_dims>::as_poly_input() {
  TorchIdxT total_size = 0;
  for (unsigned i = 0; i < point_values.size(); ++i) {
    const auto &vec = point_values[i];
    always_assert(vec.size() % constants.n_poly_point_values == 0);
    unsigned size = vec.size() / constants.n_poly_point_values;

#ifndef NDEBUG
    std::array<TorchIdxT, n_prior_dims> idx_raw;
    for (unsigned j = 0; j < n_prior_dims; ++j) {
      idx_raw[j] = i / index_multipliers[j];
    }
    auto idx = as_tensor_index(idx_raw);

    always_assert(counts.index(idx).item().template to<TorchIdxT>() ==
                  TorchIdxT(size));
#endif
    total_size += size;
  }

  at::Tensor point_values_out =
      at::empty({total_size, constants.n_poly_point_values});
  at::Tensor item_to_left_idxs = at::empty({total_size}, long_tensor_type);
  at::Tensor item_to_right_idxs = at::empty({total_size}, long_tensor_type);

  unsigned running_total = 0;
  for (unsigned i = 0; i < point_values.size(); ++i) {
    const auto &vec = point_values[i];

    unsigned size = vec.size() / constants.n_poly_point_values;
    unsigned running_total_prev = running_total;
    for (unsigned j = 0; j < size; ++j) {
      item_to_left_idxs.index_put_(
          {TorchIdxT(running_total)},
          TorchIdxT((j == 0 ? size - 1 : j - 1) + running_total_prev));
      item_to_right_idxs.index_put_(
          {TorchIdxT(running_total)},
          TorchIdxT((j + 1) % size + running_total_prev));
      for (int k = 0; k < constants.n_poly_point_values; ++k) {
        point_values_out.index_put_(
            {TorchIdxT(running_total), k},
            vec[j * unsigned(constants.n_poly_point_values) + k]);
      }
      ++running_total;
    }
  }
  always_assert(running_total == total_size);

  auto prefix_sum_counts =
      at::cat({at::tensor({TorchIdxT(0)}), counts.flatten().cumsum(0)});
  unsigned actual = prefix_sum_counts.index({-1}).item().template to<int64_t>();
  always_assert(actual == total_size);

  return {
      .point_values = point_values_out,
      .overall_features = std::move(overall_features),
      .counts = std::move(counts),
      .prefix_sum_counts = prefix_sum_counts,
      .item_to_left_idxs = item_to_left_idxs,
      .item_to_right_idxs = item_to_right_idxs,
  };
}

template struct RegionSetter<1>;
template struct RegionSetter<2>;
template struct RegionSetter<3>;
} // namespace generate_data
