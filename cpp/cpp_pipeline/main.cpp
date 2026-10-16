#include <pipeline_core/core.h>

int main()
{
    int result = pipeline_core::answer();

    return result == 42 ? 0 : 1;
}