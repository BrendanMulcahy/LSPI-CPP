/**
 * LSPI_CPP.cpp : Defines the entry point for the console application.
 *
 * Executes a series of tests of different Agents trying to solve the inverted pendulum problem.
 * Tracks the performance of each type of agent across multiple trials and prints the results.
 */

#include "stdafx.h"
#include "Agent.h"
#include "LspiAgent.h"
#include "NoopAgent.h"
#include "CleverAgent.h"
#include <vector>
#include <array>
#include <time.h>
#include "Pendulum.h"
#include <Windows.h>
#include <conio.h>
#include "TestBlas.h"
#include "math_constants.h"
#include <thrust\host_vector.h>
#include "sample.h"

using namespace std;

#define DT_CONST 0.1f
#define NUM_TRIALS 10000
#define NUM_SAMPLE_TRIALS 100
#define DISCOUNT 0.9f
#define TEST_FIRST

/**
 * Calculates the time between the two clock events. Currently is not working as expected.
 */
double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks= clock2-clock1;
	double diffms=(diffticks*10)/CLOCKS_PER_SEC;
	return diffms;
}

int _tmain(int argc, _TCHAR* argv[])
{
#ifdef TEST_FIRST
	if(!TestBlas::run_tests())
		getch();
#endif

	printf("%d", NUM_SAMPLE_TRIALS);
	srand((int)time(NULL));
	int random_agent_life = 0;
	int clever_agent_life = 0;
	int noop_agent_life = 0;
	int lspi_agent_life = 0;

	thrust::host_vector<sample> samples;
	thrust::host_vector<sample>::iterator it = samples.end(); 
	for(int i = 0; i < NUM_SAMPLE_TRIALS; i++)
	{
		Pendulum pen;
		Agent agent;
		while(!pen.isHorizontal())
		{
			// Track the random agent's samples
			sample s;
			s.angle = pen.x;
			s.angular_velocity = pen.v;

			int action = agent.getAction(pen.x, pen.v);
			pen.update(DT_CONST, action);
			float reward = pen.isHorizontal() ? -1 : 0;
			agent.notify(reward);

			s.action = action;
			s.reward = reward;
			s.final_angle = pen.x;
			s.final_angular_velocity = pen.v;

			if (reward < 0)
			{
				s.terminal = 1;
			}
			else
			{
				s.terminal = 0;
			}
			samples.insert(it, s);
			it = samples.end();
		}
	}

	clock_t start = clock();
	LspiAgent<device_vector<float>> lspi_agent(samples, DISCOUNT); 
	clock_t end = clock();
	printf("Single-threaded: %f\n", diffclock(start, end));

	for(int i = 0; i < NUM_TRIALS; i++)
	{
	/*	{
			Pendulum pen;
			Agent agent;
			while(!pen.isHorizontal())
			{
				int action = agent.getAction(pen.x, pen.v);
				pen.update(DT_CONST, action);
				random_agent_life += 1;
			}
		}

		{
			Pendulum pen;
			CleverAgent agent;
			while(!pen.isHorizontal())
			{
				int action = agent.getAction(pen.x, pen.v);
				pen.update(DT_CONST, action);
				clever_agent_life += 1;
			}
		}

		{
			Pendulum pen;
			NoopAgent agent;
			while(!pen.isHorizontal())
			{
				int action = agent.getAction(pen.x, pen.v);
				pen.update(DT_CONST, action);
				noop_agent_life += 1;
			}
		}*/

		{
			Pendulum pen;
			int temp_life = 0;
			while(!pen.isHorizontal() && temp_life < 3000)
			{
				int action = lspi_agent.getAction(pen.x, pen.v);
				pen.update(DT_CONST, action);
				lspi_agent_life += 1;
				temp_life += 1;
			}
		}
	}

	printf("\nSummary:\n");
	//printf("Random Agent: %f\n", (double)(random_agent_life*DT_CONST)/NUM_TRIALS);
	//printf("Clever Agent: %f\n", (double)(clever_agent_life*DT_CONST)/NUM_TRIALS);
	//printf("No-Op Agent: %f\n", (double)(noop_agent_life*DT_CONST)/NUM_TRIALS);
	printf("LSPI Agent: %f\n", (double)(lspi_agent_life*DT_CONST)/NUM_TRIALS);

	// Wait so we can get the results
	getch();

	return 0;
}