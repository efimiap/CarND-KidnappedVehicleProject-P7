/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

static default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  if(is_initialized){
        return;
      }
  num_particles =100;  // TODO: Set the number of particles
  /*normal_distribution<double> dist_x(gps_x, std_x);
    normal_distribution<double> dist_y(gps_y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta); 
  std --> array of uncertainties*/
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
    //std::vector<Particle> particles;
  for(int i=0; i<num_particles; i++){
    // TODO: Sample from these normal distributions like this: 
    //Add random Gaussian noise to each p
    Particle p;
    p.id = i;
    p.x =  dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  //delta_t --> the amount of time between time steps
  //std_pos --> velocity and yaw rate measured uncertainties
  // velocity and yaw rate --> current ts vel and yaw

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  for(int i =0; i<num_particles; i++){
    if (fabs(yaw_rate) <0.00001)
    {
      particles[i].x += velocity*delta_t*cos(particles[i].theta)+ dist_x(gen);
      particles[i].y += velocity*delta_t*sin(particles[i].theta) + dist_y(gen);
    }
    else{
      particles[i].x += (velocity/yaw_rate)* (sin(particles[i].theta +yaw_rate*delta_t) - sin(particles[i].theta))+ dist_x(gen);
      particles[i].y += (velocity/yaw_rate)* (-cos(particles[i].theta + yaw_rate*delta_t) +cos(particles[i].theta)) + dist_y(gen);
      particles[i].theta += yaw_rate*delta_t;
    }

    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
 //nearest neighboor data association & assign each sensor observation with the landmark ID associated with it
  for(unsigned int i=0; i< observations.size(); i++){
    
    double current_obs_x = observations[i].x;
    double current_obs_y = observations[i].y;
    double minimum_dist;
    int nearest_id;
    for(unsigned int j=0; j<predicted.size();j++){
      double current_pred_x = predicted[j].x;
      double current_pred_y = predicted[j].y;
      double distancee = dist(current_obs_x, current_obs_y, current_pred_x, current_pred_y);
      if (j==0){
        minimum_dist = distancee;
        nearest_id = predicted[j].id;
      }else{
      if(distancee < minimum_dist){
           minimum_dist =  distancee;
           nearest_id = predicted[j].id;
         }
      }
       }
    observations[i].id = nearest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  
  for(int i=0; i< num_particles; i++)
  {

    double px, py;
    px = particles[i].x;
    py = particles[i].y;
    
    vector<LandmarkObs> landmark_obs;
    for(unsigned int j=0; j<map_landmarks.landmark_list.size();j++){
      double dx = map_landmarks.landmark_list[j].x_f - px;
      double dy = map_landmarks.landmark_list[j].y_f - py;
      
      if (fabs(dx) <= sensor_range && fabs(dy) <= sensor_range){
         landmark_obs.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
      }
    }
     vector<LandmarkObs> vtm;
     for(unsigned int j = 0; j<observations.size(); j++){
       double xx = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
       double yy = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
       int idd = observations[j].id;
       vtm.push_back(LandmarkObs{idd, xx, yy});
     }

    dataAssociation(landmark_obs, vtm);
    //Calculate weights
    particles[i].weight = 1.0;

    for(unsigned int j=0; j<vtm.size(); j++){
      double os_x = vtm[j].x;
      double os_y = vtm[j].y;
      double pr_x, pr_y;
      for(unsigned int k=0; k<landmark_obs.size(); k++){
           if(landmark_obs[k].id == vtm[j].id){
              pr_x=landmark_obs[k].x;
              pr_y=landmark_obs[k].y;
           }
       }      

     double weighta = (1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp(- (pow(pr_x - os_x,2)/(2*pow(std_landmark[0],2)) + (pow(pr_y - os_y,2)/(2*pow(std_landmark[1],2)))) );
     particles[i].weight *= weighta;
      }
  }

}

void ParticleFilter::resample() {
 vector<double> weights;
  double maxW = numeric_limits<double>::min();
  
  for(int i=0; i<num_particles; i++){
    weights.push_back(particles[i].weight);
    if(particles[i].weight > maxW) {
       maxW = particles[i].weight;
    }
  }
  
  uniform_real_distribution<double> distributionDouble(0.0, maxW);
  vector<Particle> resampled;
  uniform_int_distribution<int> distribution(0, num_particles - 1);
  int index = distribution(gen);
  double beta = 0.0;
  
  for (int i=0; i<num_particles; i++){
    beta += distributionDouble(gen) * 2.0;
    while(beta > weights[index] ){
      beta = beta - weights[index];
      index = (index + 1) % num_particles;
    }
    resampled.push_back(particles[index]);
  }
  particles = resampled;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}