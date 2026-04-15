
EDV = 
weightratio = 
BW = 

Wall_Volume_LV_and_SEP = 

r_LV_and_SEP = (EDV_LV * 3 / (4* pi))^(1/3); % use eqn of sphere to get radius
r_RV         = (EDV_RV * 3 / (4* pi))^(1/3); 

h_LV_and_SEP = ( (Wall_Volume_LV_and_SEP + EDV_LV) * 3/(4*pi) ) ^ (1/3)-r_LV_and_SEP;
h_RV = h_LV_and_SEP/2; 

r_m_LV_and_SEP = r_LV_and_SEP + h_LV_and_SEP/2; 
r_m_RV         = r_RV + h_RV/2;  

r_o_LV_and_SEP = r_LV_and_SEP + h_LV_and_SEP; 
r_o_RV         = r_RV + h_RV; 

Amref_LV_and_SEP = 4 * pi * (r_m_LV_and_SEP)^2; % assume sphere 
Am_RV            = 4 * pi * (r_m_RV)^2;

Amref_LV  = Amref_LV_and_SEP * 2/3; % Assume LV is 2/3 of LV+SEP 
Amref_SEP = Amref_LV_and_SEP * 1/3; % Assume SEP is 1/3 of LV+SEP
Amref_RV  = Am_RV;

Vw_chamber_LV_and_SEP = 4/3 * pi * r_o_LV_and_SEP^3;  
Vw_chamber_RV         = 4/3 * pi * r_o_RV^3; 

Vw_LV_and_SEP = Vw_chamber_LV_and_SEP - EDV_LV; 
Vw_RV         = Vw_chamber_RV - EDV_RV;  

Vw_LV  = Vw_LV_and_SEP * 2/3; % Assume LV is 2/3 of LV+SEP 
Vw_SEP = Vw_LV_and_SEP * 1/3; % Assume SEP is 1/3 of LV+SEP 