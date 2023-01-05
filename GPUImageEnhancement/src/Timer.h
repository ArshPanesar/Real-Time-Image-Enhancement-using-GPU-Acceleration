#pragma once
#include "pch.h"

class Timer
{
protected:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
	const char* m_Title;
	bool m_Stopped;

	long long m_DurationInMicroseconds;
public:
	Timer() :
		m_Stopped(true),
		m_Title("")
	{

	}

	Timer(const char* Title) :
		m_Stopped(true),
		m_Title(Title)
	{
		Start(m_Title);
	}

	~Timer()
	{
		Stop();
	}

	void Start(const char* Title)
	{
		if (!m_Stopped)
		{
			std::cout << "Timer already Running. Cannot Start Timer with New Title: " << Title << std::endl;
			return;
		}

		m_Title = Title;
		m_StartTimePoint = std::chrono::high_resolution_clock::now();
		m_Stopped = false;
	}

	void Stop()
	{
		auto EndPoint = std::chrono::high_resolution_clock::now();
		if (m_Stopped)
			return;

		auto Start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimePoint).time_since_epoch().count();
		auto End = std::chrono::time_point_cast<std::chrono::microseconds>(EndPoint).time_since_epoch().count();

		m_DurationInMicroseconds = End - Start;

		//std::cout << "[ " << m_Title << " ] " << "Time Taken: " << m_DurationInNanoseconds << " nanoseconds" << std::endl;

		m_Stopped = true;
	}

	const long long GetDurationInMicroseconds() const
	{
		return m_DurationInMicroseconds;
	}
};
