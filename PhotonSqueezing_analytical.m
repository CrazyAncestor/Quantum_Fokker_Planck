syms x y a b c d e f h k l o p q gamma n_th D;
assume([x,y,  a, b, c, d, e, f, h, k, l, o, p, q,gamma,n_th,D], 'real')
P(x,y) =  exp(a + b * x + c * y + d* x^2 + e * y^2 + f * x * y + h * 1i + k * 1i * x + l * 1i * y + o * 1i* x^2 + p * 1i * y^2 + q * 1i * x * y);
function fx = fst_deriv(func,var)
    fx = simplify(diff(func,var));
end

function fxx = snd_deriv(func,var)
    fxx = simplify(diff(diff(func,var),var));
end

function fxy = diff_deriv(func,var1,var2)
    fxy = simplify(diff(diff(func,var1),var2));
end

function fx = exp_fst_deriv(func,var)
    fx = simplify(diff(func,var)/func);
end

function fxx = exp_snd_deriv(func,var)
    fxx = simplify(diff(diff(func,var),var)/func);
end

function fxy = exp_diff_deriv(func,var1,var2)
    fxy = simplify(diff(diff(func,var1),var2)/func);
end

%   Calculate 1st and 2nd derivatives
function [Px,Py,Pxx,Pyy,Pxy] = fst_snd_derivs(P,x,y)
    Px = exp_fst_deriv(P,x);
    Py = exp_fst_deriv(P,y);

    Pxx = exp_snd_deriv(P,x);
    Pyy = exp_snd_deriv(P,y);
    Pxy = exp_diff_deriv(P,x,y);
end

%   Extract each components in the time derivative
function [PT_const, PTx, PTy, PTxx, PTyy, PTxy] = extract_coef(PT,x,y) 
    PTxx = snd_deriv(PT,x);
    PTyy = snd_deriv(PT,y);

    PTxy = diff_deriv(PT,x,y);
    
    PTx = simplify(fst_deriv(PT,x) - (PTxx * x + PTxy * y));
    PTy = simplify(fst_deriv(PT,y) - (PTyy * y + PTxy * x));

    PT_const = PT - ((PTxx * x^2 + PTyy * y^2)/2 + (PTxy * x * y) + (PTx * x + PTy * y));
    PT_const = simplify(PT_const);

    PTxx = PTxx/2;
    PTyy = PTyy/2;
end

function [PT_constr, PTxr, PTyr,PTxxr, PTyyr, PTxyr , PT_consti, PTxi, PTyi,PTxxi, PTyyi, PTxyi] = separate_ri(PT_const, PTx, PTy,PTxx, PTyy, PTxy)
    PT_constr = simplify(real(PT_const));
    PT_consti = simplify(imag(PT_const));

    PTxr = simplify(real(PTx));
    PTxi = simplify(imag(PTx));

    PTyr = simplify(real(PTy));
    PTyi = simplify(imag(PTy));
    
    PTxxr = simplify(real(PTxx));
    PTxxi = simplify(imag(PTxx));

    PTyyr = simplify(real(PTyy));
    PTyyi = simplify(imag(PTyy));

    PTxyr = simplify(real(PTxy));
    PTxyi = simplify(imag(PTxy));
end

%   Calculate the time-derivative of P of the system
[Px,Py,Pxx,Pyy,Pxy] = fst_snd_derivs(P,x,y);

PT1 = (gamma) + (gamma/2 * x) * Px + (gamma/2 * y + 4 * D * x) * Py;
PT2 = gamma * n_th / 4 * (Pxx + Pyy) - D * Pxy;

PT = simplify(PT1 + PT2);

[PT_const, PTx, PTy,PTxx, PTyy, PTxy] = extract_coef(PT,x,y);

assume([x,y,  a, b, c, d, e, f, h, k, l, o, p, q,gamma,n_th,D], 'real')
[PT_constr, PTxr, PTyr,PTxxr, PTyyr, PTxyr , PT_consti, PTxi, PTyi,PTxxi, PTyyi, PTxyi] = separate_ri(PT_const, PTx, PTy,PTxx, PTyy, PTxy)
