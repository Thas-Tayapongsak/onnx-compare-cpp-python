namespace
{
    int my_answer() { return 42; }
} // namespace

namespace pipeline_core
{
    int answer() { return my_answer(); }
} // namespace pipeline_core