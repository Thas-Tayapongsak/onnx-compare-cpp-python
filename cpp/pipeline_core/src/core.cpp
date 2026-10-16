namespace {
    int my_answer()
    {
        return 42;
    }
}

namespace pipeline_core {
    int answer()
    {
        return my_answer();
    }
}