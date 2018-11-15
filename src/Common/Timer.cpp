#include "Common/Timer.h"
#include <iostream>
#include <iomanip>


using namespace MonocularSfM;


Timer::Timer() : started_(false), paused_(false) { }


void Timer::Start()
{
    if(started_)
        return;

    started_ = true;
    start_time_ = std::chrono::high_resolution_clock::now();
}

void Timer::Restart()
{
    started_ = false;
    Start();
}

void Timer::Pause()
{
    paused_ = true;
    pause_time_ = std::chrono::high_resolution_clock::now();
}

void Timer::Resume()
{
    if(!paused_)
        return;

    paused_ = false;

    start_time_ += std::chrono::high_resolution_clock::now() - pause_time_;

}

void Timer::Reset()
{
    started_ = false;
    paused_ = false;
}


double Timer::ElapsedMicorSeconds() const
{
    if(!started_)
        return 0.0;
    if(paused_)
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(pause_time_ - start_time_).count();
    }
    else
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time_).count();
    }
}

double Timer::ElapsedSeconds() const
{
    return ElapsedMicorSeconds() / 1e6;
}
double Timer::ElapsedMinutes() const
{
    return ElapsedSeconds() / 60;
}
double Timer::ElapsedHours() const
{
    return ElapsedMinutes() / 60;
}


void Timer::PrintSeconds() const
{
    std::cout << "Elapsed time: " << std::setiosflags(std::ios::fixed)
              << std::setprecision(5) << ElapsedSeconds() << " [seconds]"
              << std::endl;
}



void Timer::PrintMinutes() const
{
    std::cout << "Elapsed time: " << std::setiosflags(std::ios::fixed)
              << std::setprecision(5) << ElapsedMinutes() << " [minutes]"
              << std::endl;
}



void Timer::PrintHours() const
{
    std::cout << "Elapsed time: " << std::setiosflags(std::ios::fixed)
              << std::setprecision(5) << ElapsedHours() << " [hours]"
              << std::endl;
}
