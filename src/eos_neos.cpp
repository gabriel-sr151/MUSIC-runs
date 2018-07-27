// Copyright 2018 @ Chun Shen

#include "eos_neos.h"
#include "util.h"

#include <sstream>
#include <fstream>

using std::stringstream;
using std::string;

EOS_neos::EOS_neos() {
    set_EOS_id(10);
    set_number_of_tables(0);
    set_eps_max(1e5);
}


EOS_neos::~EOS_neos() {
    int ntables = get_number_of_tables();
    for (int itable = 0; itable < ntables; itable++) {
        Util::mtx_free(mu_B_tb[itable],
                       nb_length[itable], e_length[itable]);
        if (get_EOS_id() == 13) {
            Util::mtx_free(mu_S_tb[itable],
                           nb_length[itable], e_length[itable]);
        }
        if (get_EOS_id() == 14) {
            Util::mtx_free(mu_S_tb[itable],
                           nb_length[itable], e_length[itable]);
            Util::mtx_free(mu_C_tb[itable],
                           nb_length[itable], e_length[itable]);
        }
    }
}


void EOS_neos::initialize_eos(int eos_id_in) {
    // read the lattice EOS pressure, temperature, and 
    music_message.info("Using lattice EOS at finite muB from A. Monnai");
    
    set_EOS_id(eos_id_in);
    auto envPath = get_hydro_env_path();
    stringstream spath;
    spath << envPath;

    bool flag_muS = false;
    bool flag_muC = false;
    string eos_file_string_array[7];
    if (eos_id_in == 10) {
        music_message.info("reading EOS neos ...");
        spath << "/EOS/neos_2/";
        string string_tmp[] = {"0a", "0b", "0c", "1a", "2", "3", "4"};
        std::copy(std::begin(string_tmp), std::end(string_tmp),
                  std::begin(eos_file_string_array));
    } else if (eos_id_in == 11) {
        music_message.info("reading EOS neos3 ...");
        spath << "/EOS/neos_3/";
        string string_tmp[] = {"1", "2", "3", "4", "5", "6", "7"};
        std::copy(std::begin(string_tmp), std::end(string_tmp),
                  std::begin(eos_file_string_array));
    } else if (eos_id_in == 12) {
        music_message.info("reading EOS neos_b ...");
        spath << "/EOS/neos_b/";
        string string_tmp[] = {"1", "2", "3", "4", "5", "6", "7"};
        std::copy(std::begin(string_tmp), std::end(string_tmp),
                  std::begin(eos_file_string_array));
    } else if (eos_id_in == 13) {
        music_message.info("reading EOS neos_bs ...");
        spath << "/EOS/neos_bs/";
        string string_tmp[] = {"1s", "2s", "3s", "4s", "5s", "6s", "7s"};
        std::copy(std::begin(string_tmp), std::end(string_tmp),
                  std::begin(eos_file_string_array));
        flag_muS = true;
    } else if (eos_id_in == 14) {
        music_message.info("reading EOS neos_bqs ...");
        spath << "/EOS/neos_bqs/";
        string string_tmp[] = {"1qs", "2qs", "3qs", "4qs", "5qs", "6qs", "7qs"};
        std::copy(std::begin(string_tmp), std::end(string_tmp),
                  std::begin(eos_file_string_array));
        flag_muS = true;
        flag_muC = true;
    }
    
    string path = spath.str();
    music_message << "from path " << path;
    music_message.flush("info");
    
    const int ntables = 7;
    set_number_of_tables(ntables);
    resize_table_info_arrays();

    pressure_tb    = new double** [ntables];
    temperature_tb = new double** [ntables];
    mu_B_tb        = new double** [ntables];
    if (flag_muS) {
        mu_S_tb = new double** [ntables];
    }
    if (flag_muC) {
        mu_C_tb = new double** [ntables];
    }

    for (int itable = 0; itable < ntables; itable++) {
        std::ifstream eos_p(path + "neos" + eos_file_string_array[itable]
                            + "_p.dat");
        if (!eos_p.is_open()) {
            music_message << "Can not open EOS files: "
                          << (path + "neos" + eos_file_string_array[itable]
                              + "_p.dat");
            music_message.flush("error");
            exit(1);
        }
        std::ifstream eos_T(path + "neos" + eos_file_string_array[itable]
                            + "_t.dat");
        std::ifstream eos_mub(path + "neos" + eos_file_string_array[itable]
                              + "_mub.dat");
        std::ifstream eos_muS;
        std::ifstream eos_muC;
        if (flag_muS) {
            std::ifstream eos_muS(path + "neos" + eos_file_string_array[itable]
                                  + "_mus.dat");
        }
        if (flag_muC) {
            std::ifstream eos_muC(path + "neos" + eos_file_string_array[itable]
                                  + "_muq.dat");
        }
        // read the first two lines with general info:
        // first value of rhob, first value of epsilon
        // deltaRhob, deltaE, number of rhob points, number of epsilon points
        // the table size is
        // (number of rhob points + 1, number of epsilon points + 1)
        int N_e, N_rhob;
        eos_p >> nb_bounds[itable] >> e_bounds[itable];
        eos_p >> nb_spacing[itable] >> e_spacing[itable]
              >> N_rhob >> N_e;
        nb_length[itable] = N_rhob + 1;
        e_length[itable]  = N_e + 1;

        e_bounds[itable]  /= hbarc;   // 1/fm^4
        e_spacing[itable] /= hbarc;   // 1/fm^4

        // skip the header in T and mu_B files
        string dummy;
        std::getline(eos_T, dummy);
        std::getline(eos_T, dummy);
        std::getline(eos_mub, dummy);
        std::getline(eos_mub, dummy);
        if (eos_muS.is_open()) {
            std::getline(eos_muS, dummy);
            std::getline(eos_muS, dummy);
        }
        if (eos_muC.is_open()) {
            std::getline(eos_muC, dummy);
            std::getline(eos_muC, dummy);
        }

        // allocate memory for EOS arrays
        pressure_tb[itable] = Util::mtx_malloc(nb_length[itable],
                                               e_length[itable]);
        temperature_tb[itable] = Util::mtx_malloc(nb_length[itable],
                                                  e_length[itable]);
        mu_B_tb[itable] = Util::mtx_malloc(nb_length[itable],
                                           e_length[itable]);
        if (flag_muS) {
            mu_S_tb[itable] = Util::mtx_malloc(nb_length[itable],
                                               e_length[itable]);
        }
        if (flag_muC) {
            mu_C_tb[itable] = Util::mtx_malloc(nb_length[itable],
                                               e_length[itable]);
        }

        // read pressure, temperature and chemical potential values
        for (int j = 0; j < e_length[itable]; j++) {
            for (int i = 0; i < nb_length[itable]; i++) {
                eos_p >> pressure_tb[itable][i][j];
                eos_T >> temperature_tb[itable][i][j];
                eos_mub >> mu_B_tb[itable][i][j];

                if (flag_muS) {
                    eos_muS >> mu_S_tb[itable][i][j];
                    mu_S_tb[itable][i][j] /= hbarc;    // 1/fm
                }
                if (flag_muC) {
                    eos_muC >> mu_C_tb[itable][i][j];
                    mu_C_tb[itable][i][j] /= hbarc;    // 1/fm
                }

                pressure_tb[itable][i][j]    /= hbarc;    // 1/fm^4
                temperature_tb[itable][i][j] /= hbarc;    // 1/fm
                mu_B_tb[itable][i][j]        /= hbarc;    // 1/fm
            }
        }
    }
    
    //double eps_max_in = (e_bounds[6] + e_spacing[6]*e_length[6])/hbarc;
    double eps_max_in = e_bounds[6] + e_spacing[6]*e_length[6];
    set_eps_max(eps_max_in);

    music_message.info("Done reading EOS.");
}


double EOS_neos::p_e_func(double e, double rhob) const {
    return(get_dpOverde3(e, rhob));
}


double EOS_neos::p_rho_func(double e, double rhob) const {
    return(get_dpOverdrhob2(e, rhob));
}


//! This function returns the local temperature in [1/fm]
//! input local energy density eps [1/fm^4] and rhob [1/fm^3]
double EOS_neos::get_temperature(double e, double rhob) const {
    int table_idx = get_table_idx(e);
    double T = interpolate2D(e, std::abs(rhob), table_idx,
                             temperature_tb);  // 1/fm
    return(T);
}


//! This function returns the local pressure in [1/fm^4]
//! the input local energy density [1/fm^4], rhob [1/fm^3]
double EOS_neos::get_pressure(double e, double rhob) const {
    int table_idx = get_table_idx(e);
    double f = interpolate2D(e, std::abs(rhob), table_idx, pressure_tb);
    return(f);
}


//! This function returns the local baryon chemical potential  mu_B in [1/fm]
//! input local energy density eps [1/fm^4] and rhob [1/fm^3]
double EOS_neos::get_mu(double e, double rhob) const {
    int table_idx = get_table_idx(e);
    double sign = rhob/(std::abs(rhob) + 1e-15);
    double mu = sign*interpolate2D(e, std::abs(rhob), table_idx,
                                   mu_B_tb);  // 1/fm
    return(mu);
}


double EOS_neos::get_s2e(double s, double rhob) const {
    double e = get_s2e_finite_rhob(s, rhob);
    return(e);
}
