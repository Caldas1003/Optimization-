% function [] = Xango_car_metamodel_SCRIPT()
%% Run this script befor running the Xango_car_metamodel project

%% Inicializa��o
clc; clear all; close all;

%% Masses
Mc = 300;       % massa do carro [kg]
Mst = 120;       % massa suspensa traseira [kg]
Msd = 100;       % massa suspensa dianteira [kg]

%% Rigidez
Kte = 5e6;      % rigidez da suspens�o traseira esquerda [N/m]
Ktd = 5e6;      % rigidez da suspens�o traseira direita [N/m]
Kde = 3e6;      % rigidez da suspens�o dianteira esquerda [N/m]
Kdd = 3e6;      % rigidez da suspens�o dianteira direita [N/m]
Kpte = 2e6;     % rigidez do pneu traseiro esquerdo [N/m]
Kptd = 2e6;     % rigidez do pneu traseiro direito [N/m]
Kpde = 2e6;     % rigidez do pneu dianteiro esquerdo [N/m]
Kpdd = 2e6;     % rigidez do pneu dianteiro direito [N/m]
Kf = 3.01e7;    % rigidez flexional do chassis [N/m]
Kt = 1.6e7;     % rigidez torcional do chassis [N/m]

%% Amortecedores
Cte = 5e3;      % amortecimento da suspens�o traseira esquerda [N*s/m]
Ctd = 5e3;      % amortecimento da suspens�o traseira direita [N*s/m]
Cde = 3e3;      % amortecimento da suspens�o dianteira esquerda [N*s/m]
Cdd = 3e3;      % amortecimento da suspens�o dianteira direita [N*s/m]

%% In�rcias
If = 5e6;       % in�rcia flexional do chassis [kg*m^2]
It = 5e6;       % in�rcia torcional do chassis [kg*m^2]

%% Geometrias
Lt = 0.8;       % comprimento traseiro partindo do CG [m]
Ld = 1;       % comprimento dianteiro partindo do CG [m]
W2 = 0.45;      % metade do entre-rodas (bitola) [m]

%% Din�mica horizontal
Psid = 0;        %slip angle fixo

% Vari�veis Gerais
X_VL = 0; % Deslocamento Longitudinal do carro [m]
X_VT = 0; % Deslocamento Transversal do carro [m]
dot_X_VL = 0.001; % Velocidade Longitudinal do carro [m/s]
dot_X_VT = 0; % Velocidade Transversal do carro [m/s]
ddot_X_VL = 0; % Acelera��o Longitudinal do carro [m/s^2]
ddot_X_VT = 0; % Acelera��o Transversal do carro [m/s^2]

mu = 1.45; % Coeficiente de atrito [adimensional]

dot_omega_d = 0; % Velocidade Angular do pneu dianteiro esquerdo [rad/s]
dot_omega_t = 0; % Velocidade Angular do pneu traseiro esquerdo [rad/s]
ddot_omega_d = 0; % Acelera��o Angular do pneu dianteiro esquerdo [rad/s^2]
ddot_omega_t = 0; % Acelera��o Angular do pneu traseiro esquerdo [rad/s^2]

alpha = 5 * pi / 180; % Convers�o de graus para radianos
a = 1.2; % Constante de Ackermann [adimensional]
gamma = 9; % �ngulo de slip Traseiro [graus]
lambda = 9; % Slip Ratio [adimensional]

% �ngulos de C�mber
C_alpha_e = 0.5; % �ngulo de C�mber Esquerdo [graus]
C_alpha_d = 0.5; % �ngulo de C�mber Direito [graus]
C_gamma = 0; % �ngulo de C�mber Traseiro [rad]

% For�as Laterais associadas ao C�mber
C_tde = 0; % For�a Lateral Dianteira Esquerda [N]
C_tdd = 0; % For�a Lateral Dianteira Direita [N]
C_tte = 0; % For�a Lateral Traseira Esquerda [N]
C_ttd = 0; % For�a Lateral Traseira Direita [N]

Rdp = 0.3; % Raio do pneu [m]

% Massas, In�rcias e Cargas
I_p = 1.0; % In�rcia do pneu [kg.m^2]
F_zde = Mc*9.81/4; % Carga Vertical Dianteira Esquerda [N]
F_zdd = Mc*9.81/4; % Carga Vertical Dianteira Direita [N]
F_zte = Mc*9.81/4; % Carga Vertical Traseira Esquerda [N]
F_ztd = Mc*9.81/4; % Carga Vertical Traseira Direita [N]

% Entradas
tau_F = 500; % Torque de frenagem [N.m]
tau_M = 1000; % Torque Mec�nico (PWT) [N.m]


% Par�metros de Pacejka
E = 0.3336564873588197; 
C_y = 1.627;
C_x = 1.0;
C_z = 1.0;
c1 = 931.405;
c2 = 366.493;

% Par�metros adicionais (exemplos)
Ld = 1.2; % Dist�ncia do CG para dianteiro [m]
Lt = 1.2; % Dist�ncia do CG para traseiro [m]
W = 0.9; % Largura do carro [m]
H_cg = 0.3; % Altura do centro de gravidade [m]
I_psi = 1e7; % In�rcia em Yaw [kg.m^2]
I_rho = 1e7; % In�rcia em Rolagem [kg.m^2]
I_beta = 1e7; % In�rcia em Pitch [kg.m^2]
M_total = 300; % Massa total do ve�culo [kg]

%% 2. C�lculo dos Par�metros Intermedi�rios

% C_s em fun��o da carga
C_s_de = c1 * sin(2 * atan(F_zde / c2));
C_s_dd = c1 * sin(2 * atan(F_zdd / c2));
C_s_te = c1 * sin(2 * atan(F_zte / c2));
C_s_td = c1 * sin(2 * atan(F_ztd / c2));

% D em fun��o da carga
D_de = mu * F_zde;
D_dd = mu * F_zdd;
D_te = mu * F_zte;
D_td = mu * F_ztd;

% B_x, B_y e B_z em fun��o da carga
B_z_de = C_s_de / (C_z * D_de);
B_x_de = C_s_de / (C_x * D_de);
B_y_de = C_s_de / (C_y * D_de);

B_z_dd = C_s_dd / (C_z * D_dd);
B_x_dd = C_s_dd / (C_x * D_dd);
B_y_dd = C_s_dd / (C_y * D_dd);

B_z_te = C_s_te / (C_z * D_te);
B_x_te = C_s_te / (C_x * D_te);
B_y_te = C_s_te / (C_y * D_te);

B_z_td = C_s_td / (C_z * D_td);
B_x_td = C_s_td / (C_x * D_td);
B_y_td = C_s_td / (C_y * D_td);

%% 3. C�lculo das For�as e Momentos nas Quatro Rodas

% C�lculo das acelera��es angulares dos pneus
ddot_omega_d = -tau_F / I_p;
ddot_omega_t = (tau_M - tau_F) / I_p;

% C�lculo do �ngulo de Ackermann
alpha_a = alpha * a;

% C�lculo do Slip Ratio
lambda_d = (Rdp * dot_omega_d / dot_X_VL) - 1;
lambda_t = (Rdp * dot_omega_t / dot_X_VL) - 1;

% For�as Laterais
F_yde = D_de * sin(C_y * atan(B_y_de * alpha - E * (B_y_de * alpha - atan(B_y_de * alpha)))) + C_tde;
F_ydd = D_dd * sin(C_y * atan(B_y_dd * alpha_a - E * (B_y_dd * alpha_a - atan(B_y_dd * alpha_a)))) + C_tdd;
F_yte = D_te * sin(C_y * atan(B_y_te * gamma - E * (B_y_te * gamma - atan(B_y_te * gamma)))) + C_tte;
F_ytd = D_td * sin(C_y * atan(B_y_td * gamma - E * (B_y_td * gamma - atan(B_y_td * gamma)))) + C_ttd;

% For�as Longitudinais
F_xde = D_de * sin(C_x * atan(9 * B_x_de * lambda_de - E * (B_x_de * lambda_de - atan(B_x_de * lambda_de))));
F_xdd = D_dd * sin(C_x * atan(9 * B_x_dd * lambda_dd - E * (B_x_dd * lambda_dd - atan(B_x_dd * lambda_dd))));
F_xte = D_te * sin(C_x * atan(9 * B_x_te * lambda_te - E * (B_x_te * lambda_te - atan(B_x_te * lambda_te))));
F_xtd = D_td * sin(C_x * atan(9 * B_x_td * lambda_td - E * (B_x_td * lambda_td - atan(B_x_td * lambda_td))));

% Momentos Auto-alinhantes
M_zde = (D_de / 55) * sin(C_z * atan(B_z_de * alpha - E * (B_z_de * alpha - atan(B_z_de * alpha)))) + 10;
M_zdd = (D_dd / 55) * sin(C_z * atan(B_z_dd * alpha_a - E * (B_z_dd * alpha_a - atan(B_z_dd * alpha_a)))) + 10;
M_zte = (D_te / 55) * sin(C_z * atan(B_z_te * gamma - E * (B_z_te * gamma - atan(B_z_te * gamma)))) + 10;
M_ztd = (D_td / 55) * sin(C_z * atan(B_z_td * gamma - E * (B_z_td * gamma - atan(B_z_td * gamma)))) + 10;

% Camber Thrust nas quatro rodas
C_tde_calc = (D_de / 2) * sin(C_y * atan(B_y_de * C_alpha_de));
C_tdd_calc = (D_dd / 2) * sin(C_y * atan(B_y_dd * C_alpha_dd));
C_tte_calc = (D_te / 2) * sin(C_y * atan(B_y_te * C_gamma));
C_ttd_calc = (D_td / 2) * sin(C_y * atan(B_y_td * C_gamma));

%% 4. C�lculo dos Momentos Totais

% Momento de Yaw (M_psi)
M_psi = (F_yde + F_ydd) * Ld - (F_yte + F_ytd) * Lt + (F_xde - F_xdd + F_xte - F_xtd) * W;

% Momento de Rolagem (M_rho)
M_rho = (F_yde + F_ydd + F_yte + F_ytd) * H_cg;

% Momento de Pitch (M_beta)
M_beta = (F_xde + F_xdd + F_xte + F_xtd) * H_cg;

%% 5. C�lculo das Acelera��es Angulares

ddot_psi = M_psi / I_psi; % Acelera��o Angular de Yaw [rad/s^2]
ddot_rho = M_rho / I_rho; % Acelera��o Angular de Rolagem [rad/s^2]
ddot_beta = M_beta / I_beta; % Acelera��o Angular de Pitch [rad/s^2]

%% 6. C�lculo das Acelera��es Radiais

% Acelera��o Radial Longitudinal
sum_Fx = F_xde + F_xdd + F_xte + F_xtd;
ddot_X_VL = sum_Fx / M_total;

% Acelera��o Radial Transversal
sum_Fy = F_yde + F_ydd + F_yte + F_ytd;
ddot_X_VT = sum_Fy / M_total;

%% 7. Exibi��o dos Resultados

disp('===== Resultados =====');
fprintf('Acelera��o Longitudinal (ddot_X_VL): %.4f m/s^2\n', ddot_X_VL);
fprintf('Acelera��o Transversal (ddot_X_VT): %.4f m/s^2\n', ddot_X_VT);
fprintf('Acelera��o Angular de Yaw (ddot_psi): %.4f rad/s^2\n', ddot_psi);
fprintf('Acelera��o Angular de Rolagem (ddot_rho): %.4f rad/s^2\n', ddot_rho);
fprintf('Acelera��o Angular de Pitch (ddot_beta): %.4f rad/s^2\n', ddot_beta);














