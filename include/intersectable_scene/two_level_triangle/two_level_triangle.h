
// TODO
#if 0
  // TODO: consider eventually freeing...
  template <intersect::accel::enum_accel::AccelType type>
  class StoredTriangleAccels {
  public:
    using Triangle = intersect::Triangle;
    using Accel = intersect::accel::AccelT<type, execution_model, Triangle>;
    using Settings = intersect::accel::AccelSettings<type>;
    using RefType = typename Accel::Ref;

    void reset() {
      free_indexes_.clear();
      for (unsigned i = 0; i < accels_.size(); i++) {
        free_indexes_.insert(i);
      }
    }

    Optional<RefType> query(const std::string &mesh_identifier,
                                    const Settings &new_settings) {
      auto it = existing_triangle_accel_vals_.find(mesh_identifier);
      if (it == existing_triangle_accel_vals_.end()) {
        return nullopt_value;
      }

      auto [index, settings] = it->second;

      if (settings != new_settings) {
        return nullopt_value;
      }

      free_indexes_.erase(index);

      return refs_[index];
    }

    RefType add(Span<const Triangle> triangles, unsigned start, unsigned end,
                const intersect::accel::AABB &aabb, const Settings &settings,
                const std::string &mesh_identifier) {
      // SPEED: try to get item which is closest in size...
      auto generate_new = [&](unsigned idx) {
        existing_triangle_accel_vals_.insert(
            std::make_pair(mesh_identifier, std::make_tuple(idx, settings)));
        return accels_[idx].gen(settings, triangles, start, end, aabb);
      };

      if (free_indexes_.empty()) {
        unsigned new_idx = accels_.size();
        accels_.push_back(Accel());
        auto new_ref = generate_new(new_idx);
        refs_.push_back(new_ref);

        return new_ref;
      } else {
        auto it = free_indexes_.begin();
        auto new_ref = generate_new(*it);
        refs_[*it] = new_ref;
        free_indexes_.erase(it);

        return new_ref;
      }
    }

  private:
    std::set<unsigned> free_indexes_;
    std::map<std::string, std::tuple<unsigned, Settings>>
        existing_triangle_accel_vals_;

    HostVector<RefType> refs_;
    HostVector<Accel> accels_;
  };

#endif
