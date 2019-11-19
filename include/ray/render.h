#pragma once

#include "BGRA.h"
#include "scene/scene.h"

namespace ray {
void render(const scene::Scene &scene, BGRA *pixels, unsigned width,
            unsigned height, const scene::Transform &m_film_to_world);
}
