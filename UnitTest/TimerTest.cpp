#include "Common/Timer.h"
#include <unistd.h>
#include <cassert>
using namespace MonocularSfM;
int main()
{
    Timer timer;
    timer.Start();;
    sleep(1);
    timer.Pause();
    sleep(1);
    timer.Resume();
    sleep(1);
    timer.Pause();
    sleep(1);
    timer.Resume();
    sleep(1);


    timer.PrintSeconds();

//    assert(timer.ElapsedSeconds() - 1 < 0.1);
}
