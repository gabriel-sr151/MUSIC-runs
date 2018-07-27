// Copyright 2018 @ Chun Shen

#ifndef SRC_EOS_BASE_H_
#define SRC_EOS_BASE_H_

#include "pretty_ostream.h"

#include <string>
#include <vector>

class EOS_base {
 private:
    int whichEOS;
    int number_of_tables;
    double eps_max;

 public:
    pretty_ostream music_message;
    std::vector<double> nb_bounds;
    std::vector<double> e_bounds;

    std::vector<double> nb_spacing;
    std::vector<double> e_spacing;

    std::vector<int> nb_length;
    std::vector<int> e_length;

    double ***pressure_tb;
    double ***temperature_tb;
    double ***mu_B_tb;
    double ***mu_S_tb;
    double ***mu_C_tb;

    EOS_base() = default;
    ~EOS_base();

    std::string get_hydro_env_path() const;

    void set_number_of_tables(int ntables) {number_of_tables = ntables;}
    int  get_number_of_tables() const {return(number_of_tables);}
    void resize_table_info_arrays();

    void set_EOS_id(int eos_id) {whichEOS = eos_id;}
    int  get_EOS_id() const {return(whichEOS);}

    // returns maximum local energy density of the EoS table
    // in the unit of [1/fm^4]
    void   set_eps_max(double eps_max_in) {eps_max = eps_max_in;}
    double get_eps_max() const {return(eps_max);}

    double interpolate1D(double e, int table_idx, double ***table) const;
    double interpolate2D(double e, double rhob, int table_idx, double ***table) const;

    int    get_table_idx(double e) const;
    double get_entropy  (double epsilon, double rhob) const;

    double calculate_velocity_of_sound_sq(double e, double rhob) const;
    double get_dpOverde3(double e, double rhob) const;
    double get_dpOverdrhob2(double e, double rhob) const;
    double get_s2e_finite_rhob(double s, double rhob) const;

    virtual void   initialize_eos () {}
    virtual double get_cs2        (double e, double rhob) const;
    virtual double p_rho_func     (double e, double rhob) const {return(0.0);}
    virtual double p_e_func       (double e, double rhob) const {return(0.0);}
    virtual double get_temperature(double epsilon, double rhob) const {return(0.0);}
    virtual double get_mu         (double epsilon, double rhob) const {return(0.0);}
    virtual double get_muS        (double epsilon, double rhob) const {return(0.0);}
    virtual double get_pressure   (double epsilon, double rhob) const {return(0.0);}
    virtual double get_s2e        (double s, double rhob) const {return(0.0);}
    virtual void   check_eos      () const {}
    
    void check_eos_with_finite_muB() const;
    void check_eos_no_muB() const;
};


#endif  // SRC_EOS_BASE_H_
