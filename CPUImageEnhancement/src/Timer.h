#pragma once
#include <chrono>
#include <iostream>

class Timer
{
protected:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
	const char* m_Title;
	bool m_Stopped;
public:
	Timer() :
		m_Stopped(false),
		m_Title("")
	{
		m_StartTimePoint = std::chrono::high_resolution_clock::now();
	}

	Timer(const char* Title) :
		m_Stopped(false),
		m_Title(Title)
	{
		m_StartTimePoint = std::chrono::high_resolution_clock::now();
	}

	~Timer()
	{
		Stop();
	}

	void Stop()
	{
		auto EndPoint = std::chrono::high_resolution_clock::now();
		if (m_Stopped)
			return;

		auto Start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimePoint).time_since_epoch().count();
		auto End = std::chrono::time_point_cast<std::chrono::microseconds>(EndPoint).time_since_epoch().count();

		auto DurationInMicroseconds = End - Start;
		double DurationInNanoseconds = DurationInMicroseconds * 0.001;

		std::cout << "[ " << m_Title << " ] " << "Time Taken: " << DurationInMicroseconds << " microseconds - " <<
			DurationInNanoseconds << " nanoseconds" << std::endl;

		m_Stopped = true;
	}
};
