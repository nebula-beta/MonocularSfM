#ifndef __TIMER_H__
#define __TIMER_H__

#include <chrono>

namespace MonocularSfM
{

class Timer
{
public:
    Timer();
    void Start();
    void Restart();
    void Pause();
    void Resume();
    void Reset();

    double ElapsedMicorSeconds() const;
    double ElapsedSeconds() const;
    double ElapsedMinutes() const;
    double ElapsedHours() const;


    void PrintSeconds() const;
    void PrintMinutes() const;
    void PrintHours() const;


private:
    bool started_;
    bool paused_;

    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point pause_time_;
};


} // namespace Undefine

#endif // __TIMER_H__
