#ifndef PENDULUM_H
#define PENDULUM_H

class Pendulum
{
	public:
		double x, v;
		Pendulum();
		bool isHorizontal();
		void update(double dt, double u);
};

#endif