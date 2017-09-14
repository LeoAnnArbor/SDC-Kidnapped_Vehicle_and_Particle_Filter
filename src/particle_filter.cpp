/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

// declare a random engine
static default_random_engine seed;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// set the number of particles
	num_particles = 50;

	// define normal distribustion for initial sensor noise
	normal_distribution<double> x_init(0, std[0]);
	normal_distribution<double> y_init(0, std[1]);
	normal_distribution<double> theta_init(0, std[2]);

	// initialize particles

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = x + x_init(seed);
		p.y = y + y_init(seed);
		p.theta = theta + theta_init(seed);
		p.weight = 1.0;

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// define normal distribustion for  sensor noise
	normal_distribution<double> x_noise(0, std_pos[0]);
	normal_distribution<double> y_noise(0, std_pos[1]);
	normal_distribution<double> theta_noise(0, std_pos[2]);

	//Determine the new state based on kinematics model

	for (int i = 0; i < num_particles; i++) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		// if theta_dot is zero
		if (fabs(yaw_rate) < 0.0001) {
			x += velocity * delta_t  * cos(theta);
			y += velocity * delta_t  * sin(theta);
		} 
		// if theta dot is nonzero
		else {

			x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
			theta += yaw_rate * delta_t;
		}

		// add noise
		particles[i].x = x + x_noise(seed);
		particles[i].y = y + y_noise(seed);
		particles[i].theta = theta + theta_noise(seed);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++) {
		// current observation
		LandmarkObs ob = observations[i];

		//initialize minimum distance
		double min_dist = numeric_limits<double>::max();

		// init id of landmark
		int map_id = -1;

		for (unsigned int j = 0; j < predicted.size(); j++) {

			//current prediction
			LandmarkObs pred = predicted[j];

			//calculate distance between observation and predicted landmarks
			double cur_dist = dist(ob.x, ob.y, pred.x, pred.y);

			//update the id of lanmark to be the smallest distance between predicted and observed landmarks
			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				map_id = pred.id;
			}

		}

		observations[i].id = map_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0 ;i < num_particles; i++) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		//place holders for landmark locations to be within sensor range
		vector<LandmarkObs> predictions;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			float x_landmark = map_landmarks.landmark_list[j].x_f;
			float y_landmark = map_landmarks.landmark_list[j].y_f;
			int id_landmark = map_landmarks.landmark_list[j].id_i;

			//only the landmarks within senor range are considered
			if (fabs(x_landmark - x) <= sensor_range && fabs(y_landmark - y) <= sensor_range) {
				predictions.push_back(LandmarkObs{id_landmark, x_landmark, y_landmark});
			}
		}

		//place holders for observed landmark locations in map coordinate
		vector<LandmarkObs> ob_trans;
		for (unsigned int j = 0; j < observations.size(); j++) {
			double x_ob_trans = cos(theta) * observations[j].x - sin(theta) * observations[j].y + x;
			double y_ob_trans = sin(theta) * observations[j].x + cos(theta) * observations[j].y + y;
			ob_trans.push_back(LandmarkObs{observations[j].id, x_ob_trans, y_ob_trans});
		}
		// perform adta association
		dataAssociation(predictions, ob_trans);

		//initialize weight
		particles[i].weight = 1.0;

		for (unsigned int j = 0; j < ob_trans.size(); j ++) {
			double x_ob, y_ob, x_pred, y_pred;

			x_ob = ob_trans[j].x;
			y_ob = ob_trans[j].y;

			int id_associated = ob_trans[j].id;

			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == id_associated) {
					x_pred = predictions[k].x;
					y_pred = predictions[k].y;
				}
			}

			// calculate weight for this observation
			double x_std = std_landmark[0];
			double y_std = std_landmark[1];

			double weight_ob = (1/(2*M_PI*x_std*y_std)) * exp(-(pow(x_pred-x_ob,2)/(2*pow(x_std, 2)) + (pow(y_pred-y_ob,2)/(2*pow(y_std, 2)))));

			// total weight is the product
			particles[i].weight *= weight_ob;
		}

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	//resample using reshampling wheel
	vector<Particle> particles_temp;
	vector<double> weights;

	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}
	// generate random starting index
  	uniform_int_distribution<int> index_rand(0, num_particles-1);
  	auto index = index_rand(seed);

  	// get max weight
  	double max_weight = *max_element(weights.begin(), weights.end());

  	// uniform random distribution [0.0, 2 * max_weight)
  	uniform_real_distribution<double> index_weight_rand(0.0, 2* max_weight);

  	double beta = 0.0;

  	for (int i = 0; i < num_particles; i++) {
    	beta += index_weight_rand(seed);

    	while (beta > weights[index]) {
      		beta -= weights[index];
      		index = (index + 1) % num_particles;
    	}

    	particles_temp.push_back(particles[index]);
  	}

  particles = particles_temp;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
