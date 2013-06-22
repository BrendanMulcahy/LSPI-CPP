/**
 * LSPI_CPP.cpp : Defines the entry point for the console application.
 *
 * Executes a series of tests of different Agents trying to solve the inverted pendulum problem.
 * Tracks the performance of each type of agent across multiple trials and prints the results.
 */

 
#define PENDULUM

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
#include <string.h>
#include <stdlib.h>
#include <fstream>
//#include "GradientAgent.h"
#include "QuakeDefs.h"
#include <iomanip>


using namespace std;

#define DT_CONST 0.1f
#define NUM_TRIALS 1000
#define NUM_SAMPLE_TRIALS 100000
#define DISCOUNT 0.95f

//#define TEST_FIRST
//#define USE_FILE // If defined, samples will be pulled from a file titled samples.txt instead of from randomly generated input

/**
 * Calculates the time between the two clock events. Currently is not working as expected.
 */
double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks= clock2-clock1;
	double diffms=(diffticks*10)/CLOCKS_PER_SEC;
	return diffms;
}

#ifdef PENDULUM
inline void getSamplesFromFile(string filename, thrust::host_vector<sample>& samples)
{
	ifstream file(filename);
	string value;
	thrust::host_vector<sample>::iterator it = samples.end(); 
	while(file.good())
	{
		sample s;
		getline(file, value, ',');
		s.angle = (float)atof(value.c_str());

		getline(file, value, ',');
		s.angular_velocity = (float)atof(value.c_str());

		getline(file, value, ',');
		s.action = atoi(value.c_str());

		getline(file, value, ',');
		s.reward = atoi(value.c_str());

		getline(file, value, ',');
		s.final_angle = (float)atof(value.c_str());

		getline(file, value, ',');
		s.final_angular_velocity = (float)atof(value.c_str());

		getline(file, value, '\n');
		s.terminal = atoi(value.c_str());

		samples.insert(it, s);
		it = samples.end();
	}
}
#endif

#ifdef PENDULUM
int _tmain(int, _TCHAR*)
{
#ifdef TEST_FIRST
	if(!TestBlas::run_tests())
		getch();
#endif

	cublasStatus_t stat = cublasCreate(&blas::handle);
	if(stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("CUBLAS Init Failure.");
		return -1;
	}

	printf("%d", NUM_SAMPLE_TRIALS);
	srand((unsigned int)time(NULL));
	int random_agent_life = 0;
	int clever_agent_life = 0;
	int noop_agent_life = 0;
	int lspi_agent_life = 0;

	thrust::host_vector<sample> samples;

#if defined(USE_FILE)
	getSamplesFromFile("D:\\Users\\Ithiel\\Documents\\College\\Thesis\\samples.txt", samples);
#else
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
			int reward = pen.isHorizontal() ? -1 : 0;
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
#endif

	clock_t start = clock();
	LspiAgent<host_vector<float>> lspi_agent(samples, DISCOUNT);
//	LspiAgent<device_vector<float>> lspi_agent(samples, DISCOUNT); 
	clock_t end = clock();
	printf("Single-threaded: %f\n", diffclock(start, end));

	for(int i = 0; i < NUM_TRIALS; i++)
	{
		{
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
		}

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
	printf("Random Agent: %f\n", (double)(random_agent_life*DT_CONST)/NUM_TRIALS);
	printf("Clever Agent: %f\n", (double)(clever_agent_life*DT_CONST)/NUM_TRIALS);
	printf("No-Op Agent: %f\n", (double)(noop_agent_life*DT_CONST)/NUM_TRIALS);
	printf("LSPI Agent: %f\n", (double)(lspi_agent_life*DT_CONST)/NUM_TRIALS);

	// Wait so we can get the results
	getch();

	return 0;
}
#else

void updatePolicy()
{
	// Load the samples into a vector and update LSPI agent's policy
	char *fname = "samples.dat";
	host_vector<sample> samples;
	string value;
	ifstream file(fname);

	thrust::host_vector<sample>::iterator it = samples.end(); 
	while(file.good())
	{
		sample s;
		lspi_action_basis_t *state = (lspi_action_basis_t*)malloc(sizeof(lspi_action_basis_t));
		lspi_action_basis_t *fstate = (lspi_action_basis_t*)malloc(sizeof(lspi_action_basis_t));
		s.state = state;
		s.final_state = fstate;

		//// Action ////
		if(!getline(file, value, ','))
		{
			break;
		}
		s.action = atoi(value.c_str());
		////////////////

		/***** START READING STATE *****/

		//// For calculated reward ////
		getline(file, value, ',');
		state->kill_diff = atoi(value.c_str());

		getline(file, value, ',');
		state->death_diff = atoi(value.c_str());

		getline(file, value, ',');
		state->health_diff = atoi(value.c_str());

		getline(file, value, ',');
		state->armor_diff = atoi(value.c_str());

		getline(file, value, ',');
		state->hit_count_diff = atoi(value.c_str());
		///////////////////////////////

		//// Stats ////
		getline(file, value, ',');
		state->stat_health = atoi(value.c_str());

		getline(file, value, ',');
		state->stat_armor = atoi(value.c_str());

		getline(file, value, ',');
		state->stat_max_health = atoi(value.c_str());
		///////////////

		//// Powerups ////
		getline(file, value, ',');
		state->pw_quad = atoi(value.c_str());

		getline(file, value, ',');
		state->pw_battlesuit = atoi(value.c_str());

		getline(file, value, ',');
		state->pw_haste = atoi(value.c_str());

		getline(file, value, ',');
		state->pw_invis = atoi(value.c_str());

		getline(file, value, ',');
		state->pw_regen = atoi(value.c_str());

		getline(file, value, ',');
		state->pw_flight = atoi(value.c_str());

		getline(file, value, ',');
		state->pw_scout = atoi(value.c_str());

		getline(file, value, ',');
		state->pw_guard = atoi(value.c_str());

		getline(file, value, ',');
		state->pw_doubler = atoi(value.c_str());

		getline(file, value, ',');
		state->pw_ammoregen = atoi(value.c_str());

		getline(file, value, ',');
		state->pw_invulnerability = atoi(value.c_str());
		//////////////////

		//// Ammo ////
		getline(file, value, ',');
		state->wp_gauntlet = atoi(value.c_str());

		getline(file, value, ',');
		state->wp_machinegun = atoi(value.c_str());
		
		getline(file, value, ',');
		state->wp_shotgun = atoi(value.c_str());
		
		getline(file, value, ',');
		state->wp_grenade_launcher = atoi(value.c_str());
		
		getline(file, value, ',');
		state->wp_rocket_launcher = atoi(value.c_str());
		
		getline(file, value, ',');
		state->wp_lightning = atoi(value.c_str());
		
		getline(file, value, ',');
		state->wp_railgun = atoi(value.c_str());
		
		getline(file, value, ',');
		state->wp_plasmagun = atoi(value.c_str());
		
		getline(file, value, ',');
		state->wp_bfg = atoi(value.c_str());
		
		getline(file, value, ',');
		state->wp_grappling_hook = atoi(value.c_str());
		//////////////

		//// Enemy Info ////
		getline(file, value, ',');
		state->enemy = atoi(value.c_str());

		getline(file, value, ',');
		state->enemy_line_dist = (float)atof(value.c_str());

		getline(file, value, ',');
		state->enemyposition_time = (float)atof(value.c_str());

		getline(file, value, ',');
		state->enemy_is_invisible = atoi(value.c_str());

		getline(file, value, ',');
		state->enemy_is_shooting = atoi(value.c_str());

		getline(file, value, ',');
		state->enemy_weapon = atoi(value.c_str());
		////////////////////

		//// Goal Info////
		getline(file, value, ',');
		state->goal_flags = atoi(value.c_str());

		getline(file, value, ',');
		state->item_type = atoi(value.c_str());
		//////////////////

		//// Exit Information ////
		getline(file, value, ',');
		state->last_enemy_area_exits = atoi(value.c_str());

		getline(file, value, ',');
		state->goal_area_exits = atoi(value.c_str());

		getline(file, value, ',');
		state->current_area_exits = atoi(value.c_str());
		//////////////////////////
		
		//// Area Numbers ////
		getline(file, value, ',');
		state->current_area_num = atoi(value.c_str());

		getline(file, value, ',');
		state->goal_area_num = atoi(value.c_str());

		getline(file, value, ',');
		state->enemy_area_num = atoi(value.c_str());
		//////////////////////////

		//// Misc ////
		getline(file, value, ',');
		state->tfl = atoi(value.c_str());

		getline(file, value, ',');
		state->last_hit_count = atoi(value.c_str());
		//////////////
		
		/***** END READING STATE *****/

		/***** START READING FINAL STATE *****/

		//// For calculated reward ////
		getline(file, value, ',');
		fstate->kill_diff = atoi(value.c_str());

		getline(file, value, ',');
		fstate->death_diff = atoi(value.c_str());

		getline(file, value, ',');
		fstate->health_diff = atoi(value.c_str());

		getline(file, value, ',');
		fstate->armor_diff = atoi(value.c_str());

		getline(file, value, ',');
		state->hit_count_diff = atoi(value.c_str());
		///////////////////////////////

		//// Stats ////
		getline(file, value, ',');
		fstate->stat_health = atoi(value.c_str());

		getline(file, value, ',');
		fstate->stat_armor = atoi(value.c_str());

		getline(file, value, ',');
		fstate->stat_max_health = atoi(value.c_str());
		///////////////

		//// Powerups ////
		getline(file, value, ',');
		fstate->pw_quad = atoi(value.c_str());

		getline(file, value, ',');
		fstate->pw_battlesuit = atoi(value.c_str());

		getline(file, value, ',');
		fstate->pw_haste = atoi(value.c_str());

		getline(file, value, ',');
		fstate->pw_invis = atoi(value.c_str());

		getline(file, value, ',');
		fstate->pw_regen = atoi(value.c_str());

		getline(file, value, ',');
		fstate->pw_flight = atoi(value.c_str());

		getline(file, value, ',');
		fstate->pw_scout = atoi(value.c_str());

		getline(file, value, ',');
		fstate->pw_guard = atoi(value.c_str());

		getline(file, value, ',');
		fstate->pw_doubler = atoi(value.c_str());

		getline(file, value, ',');
		fstate->pw_ammoregen = atoi(value.c_str());

		getline(file, value, ',');
		fstate->pw_invulnerability = atoi(value.c_str());
		//////////////////

		//// Ammo ////
		getline(file, value, ',');
		fstate->wp_gauntlet = atoi(value.c_str());

		getline(file, value, ',');
		fstate->wp_machinegun = atoi(value.c_str());
		
		getline(file, value, ',');
		fstate->wp_shotgun = atoi(value.c_str());
		
		getline(file, value, ',');
		fstate->wp_grenade_launcher = atoi(value.c_str());
		
		getline(file, value, ',');
		fstate->wp_rocket_launcher = atoi(value.c_str());
		
		getline(file, value, ',');
		fstate->wp_lightning = atoi(value.c_str());
		
		getline(file, value, ',');
		fstate->wp_railgun = atoi(value.c_str());
		
		getline(file, value, ',');
		fstate->wp_plasmagun = atoi(value.c_str());
		
		getline(file, value, ',');
		fstate->wp_bfg = atoi(value.c_str());
		
		getline(file, value, ',');
		fstate->wp_grappling_hook = atoi(value.c_str());
		//////////////

		//// Enemy Info ////
		getline(file, value, ',');
		fstate->enemy = atoi(value.c_str());

		getline(file, value, ',');
		fstate->enemy_line_dist = (float)atof(value.c_str());

		getline(file, value, ',');
		fstate->enemyposition_time = (float)atof(value.c_str());

		getline(file, value, ',');
		fstate->enemy_is_invisible = atoi(value.c_str());

		getline(file, value, ',');
		fstate->enemy_is_shooting = atoi(value.c_str());

		getline(file, value, ',');
		fstate->enemy_weapon = atoi(value.c_str());
		////////////////////

		//// Goal Info////
		getline(file, value, ',');
		fstate->goal_flags = atoi(value.c_str());

		getline(file, value, ',');
		fstate->item_type = atoi(value.c_str());
		//////////////////

		//// Exit Information ////
		getline(file, value, ',');
		fstate->last_enemy_area_exits = atoi(value.c_str());

		getline(file, value, ',');
		fstate->goal_area_exits = atoi(value.c_str());

		getline(file, value, ',');
		fstate->current_area_exits = atoi(value.c_str());
		//////////////////////////
		
		//// Area Numbers ////
		getline(file, value, ',');
		fstate->current_area_num = atoi(value.c_str());

		getline(file, value, ',');
		fstate->goal_area_num = atoi(value.c_str());

		getline(file, value, ',');
		fstate->enemy_area_num = atoi(value.c_str());
		//////////////////////////

		//// Misc ////
		getline(file, value, ',');
		fstate->tfl = atoi(value.c_str());

		getline(file, value, '\n');
		fstate->last_hit_count = atoi(value.c_str());
		//////////////

		/***** END READING FINAL STATE *****/

		samples.insert(it, s);
		it = samples.end();
	}
	file.close();

	LspiAgent<host_vector<float>> agent(samples, DISCOUNT);

	// Write to file
	ofstream outfile("lspi.pol");
	for(int i = 0; i < agent.w.size(); i++)
	{
		if(i + 1 == agent.w.size())
		{
			outfile << fixed << setprecision(8) << agent.w[i] << endl;
		}
		else
		{
			outfile << fixed << setprecision(8) << agent.w[i] << ",";
		}
	}
	outfile.close();

	// Free space used by samples
	for(int i = 0; i < samples.size(); i++)
	{
		free(samples[i].final_state);
		free(samples[i].state);
	}
}

/*
 * When Pendulum is not defined instead we are running tests against the Quake samples.
 * All we do is parse the samples and then write out the policy.
 */
int _tmain(int, _TCHAR*)
{
	LARGE_INTEGER li, before, after;
	double frequency;
	
	if(!QueryPerformanceFrequency(&li))
	{
		cout << "Failed to query performance frequency.";
	}
	else
	{
		frequency = li.QuadPart;
	}

	QueryPerformanceCounter(&before);
	updatePolicy();
	QueryPerformanceCounter(&after);

	// Calculate the time it took and save it to perf.dat
	double policy_update_time = (double)(after.QuadPart - before.QuadPart)/frequency;

	FILE* perfFile = fopen("perf.dat", "w");
	fprintf(perfFile, "Average Policy Update Time: %f\n", 1000.0*(policy_update_time));
	fclose(perfFile);

	cout << "Done!";

	getch();

	return 0;
}

#endif