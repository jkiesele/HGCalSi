
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

double alpha(double temp_C,double t_min, double t_0=1.){
    double temp = temp_C + 273.15;
    double a1 = 1.23e-17;
    double t1 = exp((12.9e3)/temp - 34.1);
    double a0 = -8.9e-17 + (4.6e-14)/temp;
    double beta = 3.07e-18;

    return a1 * exp(-t_min / t1) + a0 - beta * log(t_min/t_0);
}



std::vector<double> create_alpha_vector(double temp_C,double start, double end, std::vector<double> & ts){
    //assert start>end;

    std::vector<double> out(11);
    ts = std::vector<double> (11);
    double distance = start-end;
 //   std::cout << "[ ";
    for(int i=0;i<11;i++){
        double t = start - distance/10.* (double)i;
        out.at(i) = alpha(temp_C ,t);//invert for sorting convenience
        ts.at(i) = t;

 //       std::cout << out.at(i) << " (" << t <<") ";
    }
 //   std::cout << " ]" << std::endl;
    return out;
}

double find_intersect_binary_search(double temp_C,double t_min,double target_temp,double rel_epsilon=1e-6){

    double ref_alpha = alpha(temp_C,t_min);

    std::vector<double>  avec,ts;
    double tlow=0.000;
    double thigh=1;
    double tempalpha=ref_alpha;

    //get right range:
    while(tempalpha>=ref_alpha){
        thigh *= 10.;
        tempalpha=1.1*alpha(target_temp,thigh);//make sure to overshoot a bit here
      //
    }

 //   std::cout << "ending at " << tempalpha << " (" << thigh << ")" << std::endl;
  //  return 1;

    while(true){

        avec = create_alpha_vector(target_temp,thigh,tlow,ts);


        auto it = std::lower_bound(avec.begin(),avec.end(),ref_alpha);

        if(it == avec.end()){//cannot happen this is zero minutes
            std::cout << "SERIOUS ERROR IN ALPHA INTERSECTION MODULE" <<std::endl;//debug
        }

        int idx = std::distance(avec.begin(), it);
        idx--;

        double alow = avec.at(idx);
        double ahigh = avec.at(idx+1);
        tlow = ts.at(idx+1);
        thigh = ts.at(idx);

        double midalpha = (alow+ahigh)/2.;

    //    std::cout << ahigh << " ("<< tlow << "), " << alow << " ("<< thigh << ")  ref "<< ref_alpha << std::endl;

        if(fabs(midalpha-ref_alpha)/ref_alpha < rel_epsilon){
            return (tlow+thigh)/2.;
        }

        //return 1;

    }
}

double find_intersection(double temp_C,double t_min,double target_temp,double rel_epsilon=1e-6, double dummy=0.){
    return find_intersect_binary_search(temp_C,t_min,target_temp,rel_epsilon);
}



int main(){


    std::cout << alpha(0.,120) << std::endl;


    std::cout << find_intersect_binary_search(60.,1200,61.) << std::endl;

    for(int i=0;i<1000000;i++)//speed test 100k/s normal CPUs
        find_intersect_binary_search(60.,1200,61.) ;


}
