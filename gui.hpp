#pragma once

#include "memory"
#include <atomic>
#include <thread>
#include <vector>
#include <pybind11/numpy.h>
#include <mutex>

struct RenderData;
namespace imgui_rendering {
    void init();

    void render_frame();

    void cleanup();


    struct State {
        std::atomic<int> rendering = 0;
        std::shared_ptr<std::thread> t;
        std::shared_ptr<RenderData> data;
        std::atomic<int> width = 1200;
        std::atomic<int> height = 800;
        unsigned char const * img = nullptr;
        std::atomic<int> img_size = 0;
        std::mutex mut;
    };


    void stop_rendering();

    void start_rendering();
}

extern imgui_rendering::State state;

class ImguiController {
public:

    ImguiController() {
        state.rendering++;
        if (state.rendering > 0 && state.t == nullptr) {
            imgui_rendering::start_rendering();
        }

    };

    ImguiController(ImguiController &other) {
        state.rendering++;
    }

    ImguiController(ImguiController &&other) {
        state.rendering++;
    }

    ~ImguiController() {
        state.rendering--;
        if (state.rendering == 0) {
            imgui_rendering::stop_rendering();
        }
    }

    int get_width() {
        return state.width;
    }

    int get_height() {
        return state.height;
    }

    void set_img(const pybind11::array_t<uint8_t> &img) {
        std::scoped_lock lock(state.mut);

        if (state.img != nullptr) {
            delete[] state.img;
        }
        auto tmp = new unsigned char[img.size()];
        memcpy(tmp, img.data(), img.size());
        state.img = tmp; // TODO remove
        state.img_size = img.size();
    }
};