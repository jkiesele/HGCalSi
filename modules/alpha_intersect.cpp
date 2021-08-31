
#include <cmath>
#include <iostream>

double alpha(double temp_C,double t_min){
    double temp = temp_C + 273.15;
    double a1 = 1.23e-17;
    double t1 = exp((12.9e3)/temp - 34.1);
    double a0 = -8.9e-17 + (4.6e-14)/temp;
    double beta = 3.07e-18;

    return a1 * exp(-t_min / t1) + a0 - beta * log(t_min/1.);
}



double find_intersection(double temp_C,double t_min,double target_temp,double rel_epsilon=1e-5){
    double start_alpha = alpha(temp_C,t_min);
    //direction is inverted to temp difference

    double stepsize = 10.*(temp_C-target_temp);
    if(stepsize < 0 && -stepsize > t_min){
        stepsize =  -t_min /100.;
    }
    double tstep = t_min;


    double previousdiff=10000;

    size_t nsteps=0;
    while(true){
        nsteps++;
        tstep += stepsize;
        if(tstep <= 0.){
            tstep=t_min;
            stepsize /= 2.;
        }

        double thisalpha = alpha(target_temp,tstep);
        double thisdiff = thisalpha-start_alpha;

        if(nsteps> 1000000){
            return -1;
        }
        if(std::abs(thisdiff) < thisalpha*rel_epsilon && std::abs(previousdiff)<thisalpha*rel_epsilon){ //in interval
            return tstep+stepsize/2.;
        }

        if(std::abs(previousdiff)  > std::abs(thisdiff)){
            //good
        }
        else{//turn around and make finer
            stepsize *= -1.;
            stepsize /= 2.;

        }
        previousdiff = thisdiff;


    }

}



int main(){


    std::cout << alpha(75,3.2) << std::endl;


    std::cout << find_intersection(60,20,21) << std::endl;


}
