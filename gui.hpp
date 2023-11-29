#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include "mutex"
#include "memory"

struct RenderData;
namespace imgui_rendering {
    void init();

    void render_frame();

    void cleanup();


    struct State {
        std::atomic<int> rendering = 0;
        std::shared_ptr<std::thread> t;
        std::shared_ptr<RenderData> data;
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
};