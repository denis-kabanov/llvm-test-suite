// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %BE_RUN_PLACEHOLDER %t.out
//
// hip_nvidia has problems constructing queues due to `No device of requested
// type available`.
// XFAIL: hip_nvidia
//==-------- queue_old_interop.cpp - SYCL queue OpenCL interop test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

std::string get_type(const device &dev) {
  return ((dev.is_host()) ? "host"
                          : (dev.is_gpu() ? "OpenCL.GPU" : "OpenCL.CPU"));
}

void print_queue_info(const queue &q) {
  std::cout << "ID=" << std::hex
            << ((q.get_device().is_host() ||
                 q.get_context().get_platform().get_backend() !=
                     cl::sycl::backend::opencl)
                    ? nullptr
                    : q.get())
            << std::endl;
  std::cout << "queue wraps " << get_type(q.get_device()) << " device"
            << std::endl;
}
int main() {
  try {
    std::cout << "Create default queue." << std::endl;
    queue q;
    print_queue_info(q);

  } catch (device_error e) {
    std::cout << "Failed to create device for context" << std::endl;
  }

  auto devices = device::get_devices();
  device &deviceA = devices[0];
  device &deviceB = (devices.size() > 1 ? devices[1] : devices[0]);
  {
    std::cout << "move constructor" << std::endl;
    queue Queue(deviceA);
    size_t hash = std::hash<queue>()(Queue);
    queue MovedQueue(std::move(Queue));
    assert(hash == std::hash<queue>()(MovedQueue));
    assert(deviceA.is_host() == MovedQueue.is_host());
    if (!deviceA.is_host() &&
        deviceA.get_platform().get_backend() == cl::sycl::backend::opencl) {
      assert(MovedQueue.get() != nullptr);
    }
  }
  {
    std::cout << "move assignment operator" << std::endl;
    queue Queue(deviceA);
    size_t hash = std::hash<queue>()(Queue);
    queue WillMovedQueue(deviceB);
    WillMovedQueue = std::move(Queue);
    assert(hash == std::hash<queue>()(WillMovedQueue));
    assert(deviceA.is_host() == WillMovedQueue.is_host());
    if (!deviceA.is_host() &&
        deviceA.get_platform().get_backend() == cl::sycl::backend::opencl) {
      assert(WillMovedQueue.get() != nullptr);
    }
  }
  {
    std::cout << "copy constructor" << std::endl;
    queue Queue(deviceA);
    size_t hash = std::hash<queue>()(Queue);
    queue QueueCopy(Queue);
    assert(hash == std::hash<queue>()(Queue));
    assert(hash == std::hash<queue>()(QueueCopy));
    assert(Queue == QueueCopy);
    assert(Queue.is_host() == QueueCopy.is_host());
  }
  {
    std::cout << "copy assignment operator" << std::endl;
    queue Queue(deviceA);
    size_t hash = std::hash<queue>()(Queue);
    queue WillQueueCopy(deviceB);
    WillQueueCopy = Queue;
    assert(hash == std::hash<queue>()(Queue));
    assert(hash == std::hash<queue>()(WillQueueCopy));
    assert(Queue == WillQueueCopy);
    assert(Queue.is_host() == WillQueueCopy.is_host());
  }

  {
    property_list pl = {};
    queue Queue(pl);
    try {
      Queue.throw_asynchronous();
    } catch (const std::bad_function_call &e) {
      std::cout << "Default asynchronous handler call failed: " << e.what()
                << std::endl;
      throw;
    }
  }

  {
    default_selector Selector;
    device Device = Selector.select_device();
    context Context(Device);
    queue Queue(Context, Selector);
    assert(Context == Queue.get_context());
  }

  {
    context Context(deviceA);
    queue Queue(Context, deviceA);
    assert(Context == Queue.get_context());
  }

  if (devices.size() > 1) {
    bool GotException = false;
    try {
      context Context(deviceA);
      queue Queue(Context, deviceB);
      assert(Context == Queue.get_context());
    } catch (std::exception &e) {
      std::cout << "Exception check passed: " << e.what() << std::endl;
      GotException = true;
    }
    assert(GotException);
  }
}