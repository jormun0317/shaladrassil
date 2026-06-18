#include <stdio.h>
#include <fstream>

// coded by dqs + Jormun 20250317

#define N0 98
#define N1 49
#define S 6
#define P 8
float pi = 3.1415926, timer = 0, dt = 0.001, end = 2, mach = 0.3, reynold = 2000.0;
#define M 1

float legendre(float ps, int k)
{
    if (k == -1)
        return 0.0;
    else if (k == 0)
        return 1.0;
    else
        return ((2 * k - 1) * ps * legendre(ps, k - 1) - (k - 1) * legendre(ps, k - 2)) / k;
}

float degendre(float ps, int k)
{
    if (k == -1)
        return 0.0;
    else if (k == 0)
        return 0.0;
    else
        return ((2 * k - 1) * (ps * degendre(ps, k - 1) + legendre(ps, k - 1)) - (k - 1) * degendre(ps, k - 2)) / k;
}

struct cubit
{
    cubit *idx[4] = {nullptr, nullptr, nullptr, nullptr};
    int flg[4] = {0, 0, 0, 0};

    float spect[3][4][S]; // r d
    float value[4][P][P], value_x[4][P][P], value_y[4][P][P];
    float base[S][P][P], base_x[S][P][P], base_y[S][P][P];
    float test[S][P][P], test_x[S][P][P], test_y[S][P][P];
    float q[4][2], n[4][2], pos[2][P][P], vol; // quadrant normal positon volume
    float ma = mach, re = reynold;

    __device__ void clearance(float un[4], unsigned int p0, unsigned int p1, int flgidx)
    {
        if (flg[flgidx] == -1)
        {
            if (flgidx == 0)
                p1 = p0, p0 = P - 1;
            else if (flgidx == 1)
                p0 = p1, p1 = 0;
            else if (flgidx == 2)
                p1 = p0, p0 = 0;
            else if (flgidx == 3)
                p0 = p1, p1 = P - 1;
            for (int d = 0; d < 4; ++d)
                un[d] = idx[flgidx]->value[d][p0][p1];
        }
        if (flg[flgidx] == 0)
        {
            if (flgidx == 0)
                p1 = 0;
            else if (flgidx == 1)
                p0 = P - 1;
            else if (flgidx == 2)
                p1 = P - 1;
            else if (flgidx == 3)
                p0 = 0;
            for (int d = 0; d < 4; ++d)
                un[d] = idx[flgidx]->value[d][p0][p1];
        }
        else if (flg[flgidx] == 1)
            for (int d = 0; d < 4; ++d)
                un[d] = value[d][p0][p1];
        else if (flg[flgidx] == 2)
            un[0] = value[0][p0][p1], un[1] = -value[1][p0][p1], un[2] = -value[2][p0][p1], un[3] = value[3][p0][p1];
        else if (flg[flgidx] == 3)
        {
            float pres, presn;
            if (ma >= 1.0 && n[flgidx][0] > 0.0) // supersonic inflow
                un[0] = 1.0, un[1] = 1.0, un[2] = 0.0, un[3] = 1.0 / ma / ma / 0.56 + 0.5;
            else if (ma >= 1.0 && n[flgidx][0] <= 0.0) // supersonic outflow
                for (int d = 0; d < 4; ++d)
                    un[d] = value[d][p0][p1];
            else if (ma < 1.0 && n[flgidx][0] > 0.0) // subsonic inflow
                pres = 0.4 * value[3][p0][p1] - 0.2 * value[1][p0][p1] * value[1][p0][p1] / value[0][p0][p1] - 0.2 * value[2][p0][p1] * value[2][p0][p1] / value[0][p0][p1],
                presn = 0.5 * (1.0 / ma / ma / 1.4 + pres - (n[flgidx][0] * (value[1][p0][p1] / value[0][p0][p1] - 1.0) + n[flgidx][1] * (value[2][p0][p1] / value[0][p0][p1])) / ma),
                un[0] = 1.0 + (presn - 1.0 / ma / ma / 1.4) * ma * ma,
                un[1] = un[0] * (1.0 - n[flgidx][0] * (presn - 1.0 / ma / ma / 1.4) * ma),
                un[2] = un[0] * (0.0 - n[flgidx][1] * (presn - 1.0 / ma / ma / 1.4) * ma),
                un[3] = presn / 0.4 + 0.5 * (un[1] * un[1] + un[2] * un[2]) / un[0];
            else if (ma < 1.0 && n[flgidx][0] <= 0.0) // subsonic outflow
                pres = 0.4 * value[3][p0][p1] - 0.2 * value[1][p0][p1] * value[1][p0][p1] / value[0][p0][p1] - 0.2 * value[2][p0][p1] * value[2][p0][p1] / value[0][p0][p1],
                presn = 0.5 * (1.0 / ma / ma / 1.4 + pres - (n[flgidx][0] * (value[1][p0][p1] / value[0][p0][p1] - 1.0) + n[flgidx][1] * (value[2][p0][p1] / value[0][p0][p1])) / ma),
                un[0] = value[0][p0][p1] + (presn - pres) * ma * ma,
                un[1] = un[0] * (value[1][p0][p1] / value[0][p0][p1] + n[flgidx][0] * (presn - pres) * ma),
                un[2] = un[0] * (value[2][p0][p1] / value[0][p0][p1] + n[flgidx][1] * (presn - pres) * ma),
                un[3] = presn / 0.4 + 0.5 * (un[1] * un[1] + un[2] * un[2]) / un[0];
        }
        else if (flg[flgidx] == 4)
        {
            float norm_x = -n[flgidx][0] * (n[flgidx][0] * value[1][p0][p1] + n[flgidx][1] * value[2][p0][p1]),
                  norm_y = -n[flgidx][1] * (n[flgidx][0] * value[1][p0][p1] + n[flgidx][1] * value[2][p0][p1]);
            un[0] = value[0][p0][p1], un[1] = value[1][p0][p1] + 2.0 * norm_x, un[2] = value[2][p0][p1] + 2.0 * norm_y, un[3] = value[3][p0][p1];
        }
        else if (flg[flgidx] == 5) // exact
            ;
        else if (flg[flgidx] == 6)
            un[0] = 8.0, un[1] = 8.0 * 8.25 * sqrt(3.0) * 0.5, un[2] = -8.0 * 8.25 * 0.5,
            un[3] = 116.5 / 0.4 + 0.5 * 8.0 * (8.25 * sqrt(3.0) * 0.5 * 8.25 * sqrt(3.0) * 0.5 + 8.25 * 0.5 * 8.25 * 0.5);
        else if (flg[flgidx] == 7)
        {
            if (pos[0][p0][p1] < 1.0 / 6.0)
                un[0] = 8.0, un[1] = 8.0 * 8.25 * sqrt(3.0) * 0.5, un[2] = -8.0 * 8.25 * 0.5,
                un[3] = 116.5 / 0.4 + 0.5 * 8.0 * (8.25 * sqrt(3.0) * 0.5 * 8.25 * sqrt(3.0) * 0.5 + 8.25 * 0.5 * 8.25 * 0.5);
            else
                un[0] = value[0][p0][p1], un[1] = value[1][p0][p1], un[2] = -value[2][p0][p1], un[3] = value[3][p0][p1];
        }
    }

    __device__ void convection(float cx[4], float cy[4], int p0, int p1)
    {
        float u[4], pres, cxn[4], cyn[4], un[4], presn, alphax, alphay, nx, ny;
        u[0] = value[0][p0][p1], u[1] = value[1][p0][p1], u[2] = value[2][p0][p1], u[3] = value[3][p0][p1];
        pres = 0.4 * u[3] - 0.2 * u[1] * u[1] / u[0] - 0.2 * u[2] * u[2] / u[0],
        cx[0] = u[1], cx[1] = u[1] * u[1] / u[0] + pres, cx[2] = u[1] * u[2] / u[0], cx[3] = u[1] / u[0] * (u[3] + pres),
        cy[0] = u[2], cy[1] = u[1] * u[2] / u[0], cy[2] = u[2] * u[2] / u[0] + pres, cy[3] = u[2] / u[0] * (u[3] + pres);
        if (p1 == P - 1)
            clearance(un, p0, p1, 0), nx = n[0][0], ny = n[0][1];
        else if (p0 == 0)
            clearance(un, p0, p1, 1), nx = n[1][0], ny = n[1][1];
        else if (p1 == 0)
            clearance(un, p0, p1, 2), nx = n[2][0], ny = n[2][1];
        else if (p0 == P - 1)
            clearance(un, p0, p1, 3), nx = n[3][0], ny = n[3][1];
        else
            return;
        presn = 0.4 * un[3] - 0.2 * un[1] * un[1] / un[0] - 0.2 * un[2] * un[2] / un[0],
        cxn[0] = un[1], cxn[1] = un[1] * un[1] / un[0] + presn, cxn[2] = un[1] * un[2] / un[0], cxn[3] = un[1] / un[0] * (un[3] + presn),
        cyn[0] = un[2], cyn[1] = un[1] * un[2] / un[0], cyn[2] = un[2] * un[2] / un[0] + presn, cyn[3] = un[2] / un[0] * (un[3] + presn);
        alphax = max(std::fabs(u[1] / u[0]) + sqrt(1.4 * pres / u[0]), std::fabs(un[1] / un[0]) + sqrt(1.4 * presn / un[0])),
        alphay = max(std::fabs(u[2] / u[0]) + sqrt(1.4 * pres / u[0]), std::fabs(un[2] / un[0]) + sqrt(1.4 * presn / un[0]));
        for (int d = 0; d < 4; ++d)
            cx[d] = 0.5 * (cx[d] + cxn[d]) - 0.5 * alphax * nx * (u[d] - un[d]), cy[d] = 0.5 * (cy[d] + cyn[d]) - 0.5 * alphay * ny * (u[d] - un[d]);
    }

    __device__ void viscous(float vx[4], float vy[4], int p0, int p1)
    {
        float u[4], u_x[4], u_y[4], velou, velov, velou_x, velou_y, velov_x, velov_y, temp, ener, ener_x, ener_y, mu = 1.0, sgsx[4], sgsy[4], mut = 0;
        for (int i = 0; i < 4; ++i)
            u[i] = value[i][p0][p1], u_x[i] = value_x[i][p0][p1], u_y[i] = value_y[i][p0][p1];
        if (p1 == P - 1)
            clearance(u, p0, p1, 0);
        else if (p0 == 0)
            clearance(u, p0, p1, 1);
        else if (p1 == 0)
            clearance(u, p0, p1, 2);
        else if (p0 == P - 1)
            clearance(u, p0, p1, 3);
        temp = 1.4 * ma * ma * (0.4 * u[3] - 0.2 * u[1] * u[1] / u[0] - 0.2 * u[2] * u[2] / u[0]) / u[0], mu = pow(temp, 1.5) * 1.404 / (temp + 0.404);
        velou = u[1] / u[0], velov = u[2] / u[0], ener = u[3] / u[0] - 0.5 * (u[1] * u[1] + u[2] * u[2]) / u[0] / u[0],
        velou_x = (u_x[1] - u_x[0] * velou) / u[0], velou_y = (u_y[1] - u_y[0] * velou) / u[0],
        velov_x = (u_x[2] - u_x[0] * velov) / u[0], velov_y = (u_y[2] - u_y[0] * velov) / u[0],
        ener_x = (u_x[3] - 0.5 * (u_x[1] * velou + u[1] * velou_x + u_x[2] * velov + u[2] * velov_x) - u_x[0] * ener) / u[0],
        ener_y = (u_y[3] - 0.5 * (u_y[1] * velou + u[1] * velou_y + u_y[2] * velov + u[2] * velov_y) - u_y[0] * ener) / u[0];
        mut = 0.5 * value[0][p0][p1] * vol * sqrt(2.0 * velou_x * velou_x + (velou_y + velov_x) * (velou_y + velov_x) + 2.0 * velov_y * velov_y);
        vx[0] = 0.0, vx[1] = 2.0 * velou_x - 2.0 / 3.0 * (velou_x + velov_y), vx[2] = velou_y + velov_x, vx[3] = velou * vx[1] + velov * vx[2] + 2.0 * ener_x,
        vy[0] = 0.0, vy[1] = velou_y + velov_x, vy[2] = 2.0 * velov_y - 2.0 / 3.0 * (velou_x + velov_y), vy[3] = velou * vy[1] + velov * vy[2] + 2.0 * ener_y;
        sgsx[0] = 0.0, sgsx[1] = 2.0 * velou_x, sgsx[2] = velou_y + velov_x, sgsx[3] = velou * sgsx[1] + velov * sgsx[2] + 1.4 / 0.9 * ener_x,
        sgsy[0] = 0.0, sgsy[1] = velou_y + velov_x, sgsy[2] = 2.0 * velov_y, sgsy[3] = velou * sgsy[1] + velov * sgsy[2] + 1.4 / 0.9 * ener_y;
        for (int d = 1; d < 4; ++d)
            vx[d] *= mu / re, vy[d] *= mu / re, sgsx[d] *= mut / re, sgsy[d] *= mut / re;
        for (int d = 0; d < 4; ++d)
            vx[d] += sgsx[d], vy[d] += sgsy[d];
    }

    void caculation()
    {
        float l[4], ps[P], ws[P]; // position weight standard
        if (P == 4)
            ps[0] = -1.0, ps[1] = -0.57735026918963, ps[2] = +0.57735026918963, ps[3] = +1.0,
            ws[0] = 0.0, ws[1] = 1.0, ws[2] = 1.0, ws[3] = 0.0;
        else if (P == 8)
            ps[0] = -1.0, ps[1] = -0.9324695142031521, ps[2] = -0.6612093864662645, ps[3] = -0.2386191860831969, ps[4] = 0.2386191860831969, ps[5] = 0.6612093864662645, ps[6] = 0.9324695142031521, ps[7] = +1.0,
            ws[0] = 0.0, ws[1] = 0.1713244923791704, ws[2] = 0.3607615730481386, ws[3] = 0.4679139345726910, ws[4] = 0.4679139345726910, ws[5] = 0.3607615730481386, ws[6] = 0.1713244923791704, ws[7] = 0.0;
        else
            return;
        n[0][0] = q[0][1] - q[1][1], n[0][1] = q[1][0] - q[0][0], n[1][0] = q[1][1] - q[2][1], n[1][1] = q[2][0] - q[1][0],
        n[2][0] = q[2][1] - q[3][1], n[2][1] = q[3][0] - q[2][0], n[3][0] = q[3][1] - q[0][1], n[3][1] = q[0][0] - q[3][0];
        for (int i = 0; i < 4; ++i)
            l[i] = sqrt(n[i][0] * n[i][0] + n[i][1] * n[i][1]), n[i][0] /= l[i], n[i][1] /= l[i];

        float x0 = (q[0][0] - q[1][0] + q[2][0] - q[3][0]) / 4, x1 = (q[0][0] - q[1][0] - q[2][0] + q[3][0]) / 4, x2 = (q[0][0] + q[1][0] - q[2][0] - q[3][0]) / 4,
              y0 = (q[0][1] - q[1][1] + q[2][1] - q[3][1]) / 4, y1 = (q[0][1] - q[1][1] - q[2][1] + q[3][1]) / 4, y2 = (q[0][1] + q[1][1] - q[2][1] - q[3][1]) / 4,
              a = x1 * y0 - x0 * y1, b = x0 * y2 - x2 * y0, c = x1 * y2 - x2 * y1, x_xs, x_ys, y_xs, y_ys, jaco, j2, mass;

        int K = 1, pyramid = 0;
        while ((pyramid += K) < S)
            ++K;
        --K;
        vol = 0.0;
        for (int k = 0, s = 0; k <= K; ++k)
            for (int s0 = 0, s1; s1 = k - s0, s0 <= k; ++s, ++s0)
                for (int p0 = 0; p0 < P; ++p0)
                    for (int p1 = 0; p1 < P; ++p1)
                        x_xs = x0 * ps[p1] + x1, x_ys = x0 * ps[p0] + x2, y_xs = y0 * ps[p1] + y1, y_ys = y0 * ps[p0] + y2,
                        jaco = a * ps[p0] + b * ps[p1] + c, j2 = jaco * jaco, mass = (2.0 * s0 + 1) * (2.0 * s1 + 1) / 4,
                        base[s][p0][p1] = legendre(ps[p0], s0) * legendre(ps[p1], s1) * mass,
                        test[s][p0][p1] = legendre(ps[p0], s0) * legendre(ps[p1], s1) * ws[p0] * ws[p1],
                        base_x[s][p0][p1] = (degendre(ps[p0], s0) * legendre(ps[p1], s1) * y_ys + legendre(ps[p0], s0) * degendre(ps[p1], s1) * (-y_xs)) / jaco * mass,
                        base_y[s][p0][p1] = (degendre(ps[p0], s0) * legendre(ps[p1], s1) * (-x_ys) + legendre(ps[p0], s0) * degendre(ps[p1], s1) * x_xs) / jaco * mass,
                        test_x[s][p0][p1] = (degendre(ps[p0], s0) * legendre(ps[p1], s1) * jaco - legendre(ps[p0], s0) * legendre(ps[p1], s1) * a) / j2 * y_ys * ws[p0] * ws[p1] +
                                            (legendre(ps[p0], s0) * degendre(ps[p1], s1) * jaco - legendre(ps[p0], s0) * legendre(ps[p1], s1) * b) / j2 * (-y_xs) * ws[p0] * ws[p1],
                        test_y[s][p0][p1] = (degendre(ps[p0], s0) * legendre(ps[p1], s1) * jaco - legendre(ps[p0], s0) * legendre(ps[p1], s1) * a) / j2 * (-x_ys) * ws[p0] * ws[p1] +
                                            (legendre(ps[p0], s0) * degendre(ps[p1], s1) * jaco - legendre(ps[p0], s0) * legendre(ps[p1], s1) * b) / j2 * x_xs * ws[p0] * ws[p1],
                        vol += jaco * ws[p0] * ws[p1];
        vol /= (float)S;
        for (int k = 0, s = 0; k <= K; ++k)
            for (int s0 = 0, s1; s1 = k - s0, s0 <= k; ++s, ++s0)
                for (int p = 0; p < P; ++p)
                    jaco = a * ps[p] + b * ps[P - 1] + c,
                    test_x[s][p][P - 1] = legendre(ps[p], s0) * legendre(ps[P - 1], s1) / jaco * l[0] * n[0][0] * ws[p] / 2,
                    test_y[s][p][P - 1] = legendre(ps[p], s0) * legendre(ps[P - 1], s1) / jaco * l[0] * n[0][1] * ws[p] / 2,
                    jaco = a * ps[0] + b * ps[p] + c,
                    test_x[s][0][p] = legendre(ps[0], s0) * legendre(ps[p], s1) / jaco * l[1] * n[1][0] * ws[p] / 2,
                    test_y[s][0][p] = legendre(ps[0], s0) * legendre(ps[p], s1) / jaco * l[1] * n[1][1] * ws[p] / 2,
                    jaco = a * ps[p] + b * ps[0] + c,
                    test_x[s][p][0] = legendre(ps[p], s0) * legendre(ps[0], s1) / jaco * l[2] * n[2][0] * ws[p] / 2,
                    test_y[s][p][0] = legendre(ps[p], s0) * legendre(ps[0], s1) / jaco * l[2] * n[2][1] * ws[p] / 2,
                    jaco = a * ps[P - 1] + b * ps[p] + c,
                    test_x[s][P - 1][p] = legendre(ps[P - 1], s0) * legendre(ps[p], s1) / jaco * l[3] * n[3][0] * ws[p] / 2,
                    test_y[s][P - 1][p] = legendre(ps[P - 1], s0) * legendre(ps[p], s1) / jaco * l[3] * n[3][1] * ws[p] / 2;

        float x, y, xs, ys, xc = 5.0, yc = 5.0, xn, yn, rn, rho, u, v, temp, pres; // eddy value
        for (int p0 = 0; p0 < P; ++p0)
            for (int p1 = 0; p1 < P; ++p1)
                xs = ps[p0], ys = ps[p1],
                x = (q[0][0] * (1 + xs) * (1 + ys) + q[1][0] * (1 - xs) * (1 + ys) + q[2][0] * (1 - xs) * (1 - ys) + q[3][0] * (1 + xs) * (1 - ys)) / 4.0,
                y = (q[0][1] * (1 + xs) * (1 + ys) + q[1][1] * (1 - xs) * (1 + ys) + q[2][1] * (1 - xs) * (1 - ys) + q[3][1] * (1 + xs) * (1 - ys)) / 4.0,
                xn = x - xc, yn = y - yc, rn = xn * xn + yn * yn, u = 1.0 + 2.5 / pi * exp(0.5 * (1.0 - rn)) * (-yn), v = 1.0 + 2.5 / pi * exp(0.5 * (1.0 - rn)) * (+xn),
                temp = 1.0 - 10.0 / (11.2 * pi * pi) * exp(1.0 - rn), rho = pow(temp, 2.5), pres = pow(rho, 1.4), pos[0][p0][p1] = x, pos[1][p0][p1] = y,
                value[0][p0][p1] = rho, value[1][p0][p1] = rho * u, value[2][p0][p1] = rho * v, value[3][p0][p1] = 2.5 * pres + 0.5 * rho * (u * u + v * v);

        for (int p0 = 0; p0 < P; ++p0) // flow value
            for (int p1 = 0; p1 < P; ++p1)
                value[0][p0][p1] = 1.0, value[1][p0][p1] = 1.0, value[2][p0][p1] = 0.0, value[3][p0][p1] = 1.0 / ma / ma / 0.56 + 0.5;

        for (int d = 0; d < 4; ++d)
            for (int s = 0; s < S; ++s)
                spect[0][d][s] = 0;
        for (int d = 0; d < 4; ++d)
            for (int s = 0; s < S; ++s)
                for (int p0 = 0; p0 < P; ++p0)
                    for (int p1 = 0; p1 < P; ++p1)
                        spect[0][d][s] += value[d][p0][p1] * test[s][p0][p1];
        for (int d = 0; d < 4; ++d)
            for (int p0 = 0; p0 < P; ++p0)
                for (int p1 = 0; p1 < P; ++p1)
                    value[d][p0][p1] = 0, value_x[d][p0][p1] = 0, value_y[d][p0][p1] = 0;
        for (int d = 0; d < 4; ++d)
            for (int p0 = 0; p0 < P; ++p0)
                for (int p1 = 0; p1 < P; ++p1)
                    for (int s = 0; s < S; ++s)
                        value[d][p0][p1] += spect[0][d][s] * base[s][p0][p1], value_x[d][p0][p1] += spect[0][d][s] * base_x[s][p0][p1], value_y[d][p0][p1] += spect[0][d][s] * base_y[s][p0][p1];
    }
};

struct block
{
    cubit cbt[N0][N1];
    float coord[N0 + 1][N1 + 1][2];
    float trinity[M][2];
    int compose[N0][4];

    void bound(int cbtidx, int bcidx)
    {
        if (cbtidx == -1)
            for (int n0 = 0; n0 < N0; ++n0)
                for (int n1 = 0; n1 < N1; ++n1)
                    for (int i = 0; i < 4; ++i)
                        cbt[n0][n1].flg[i] = bcidx;
        else if (cbtidx == 0)
            for (int n0 = 0; n0 < N0; ++n0)
                cbt[n0][N1 - 1].flg[0] = bcidx;
        else if (cbtidx == 1)
            for (int n1 = 0; n1 < N1; ++n1)
                cbt[0][n1].flg[1] = bcidx;
        else if (cbtidx == 2)
            for (int n0 = 0; n0 < N0; ++n0)
                cbt[n0][0].flg[2] = bcidx;
        else if (cbtidx == 3)
            for (int n1 = 0; n1 < N1; ++n1)
                cbt[N0 - 1][n1].flg[3] = bcidx;
    }

    void slime(std::string file, float a0 = 0.0, float b0 = 10.0, float a1 = 0.0, float b1 = 10.0)
    {
        if (file == "cartesian")
        {
            float h0 = (b0 - a0) / N0, h1 = (b1 - a1) / N1;
            for (int n0 = 0; n0 <= N0; ++n0)
                for (int n1 = 0; n1 <= N1; ++n1)
                    coord[n0][n1][0] = a0 + n0 * h0, coord[n0][n1][1] = a1 + n1 * h1;
        }
        else
        {
            float trash;
            std::ifstream glasses(file, std::ios::in);
            glasses >> trash >> trash;
            for (int n0 = 0; n0 <= N0; ++n0)
                for (int n1 = 0; n1 <= N1; ++n1)
                    glasses >> coord[n0][n1][0] >> coord[n0][n1][1] >> trash;
        }
        for (int n0 = 0; n0 < N0; ++n0)
            for (int n1 = 0; n1 < N1; cbt[n0][n1].caculation(), ++n1)
                for (int s = 0; s < 2; ++s)
                    cbt[n0][n1].q[0][s] = coord[n0 + 1][n1 + 1][s], cbt[n0][n1].q[1][s] = coord[n0][n1 + 1][s], cbt[n0][n1].q[2][s] = coord[n0][n1][s], cbt[n0][n1].q[3][s] = coord[n0 + 1][n1][s];
    }

    void penrose(std::string file)
    {
        std::ifstream glasses(file, std::ios::in);
        float trash;
        glasses >> trash;
        for (int m = 0; m < M; ++m)
            glasses >> trinity[m][0] >> trinity[m][1] >> trash;
        int garbage;
        glasses >> garbage;
        for (int n = 0; n < N0; ++n)
            glasses >> compose[n][0] >> compose[n][1] >> compose[n][2] >> garbage, compose[n][3] = compose[n][0];

        float center[4][2];
        for (int n = 0; n < N0; ++n)
            for (int i = 0; i < 2; ++i)
                center[0][i] = (trinity[compose[n][0] - 1][i] + trinity[compose[n][1] - 1][i] + trinity[compose[n][2] - 1][i]) / 3.0,
                center[1][i] = (trinity[compose[n][0] - 1][i] + trinity[compose[n][1] - 1][i]) / 2.0,
                center[2][i] = (trinity[compose[n][1] - 1][i] + trinity[compose[n][2] - 1][i]) / 2.0,
                center[3][i] = (trinity[compose[n][2] - 1][i] + trinity[compose[n][0] - 1][i]) / 2.0,
                cbt[n][0].q[0][i] = center[0][i], cbt[n][0].q[1][i] = center[3][i], cbt[n][0].q[2][i] = trinity[compose[n][0] - 1][i], cbt[n][0].q[3][i] = center[1][i],
                cbt[n][1].q[0][i] = center[0][i], cbt[n][1].q[1][i] = center[1][i], cbt[n][1].q[2][i] = trinity[compose[n][1] - 1][i], cbt[n][1].q[3][i] = center[2][i],
                cbt[n][2].q[0][i] = center[0][i], cbt[n][2].q[1][i] = center[2][i], cbt[n][2].q[2][i] = trinity[compose[n][2] - 1][i], cbt[n][2].q[3][i] = center[3][i],
                cbt[n][0].caculation(), cbt[n][1].caculation(), cbt[n][2].caculation();
    }

    void paraview(int interval)
    {
        static int timing = -1;
        ++timing;
        if (timing % interval)
            return;
        std::string file, title = "x,y,z,u0,u1,u2,u3";
        if (timing > 99999)
            file = "view_99999.csv";
        else if (timing > 9999)
            file = "view_" + std::to_string(timing) + ".csv";
        else if (timing > 999)
            file = "view_0" + std::to_string(timing) + ".csv";
        else if (timing > 99)
            file = "view_00" + std::to_string(timing) + ".csv";
        else if (timing > 9)
            file = "view_000" + std::to_string(timing) + ".csv";
        else
            file = "view_0000" + std::to_string(timing) + ".csv";
        std::ofstream pen(file, std::ios::out | std::ios::trunc);
        pen << title << std::endl;
        for (int n1 = 0; n1 < N1; ++n1)
            for (int p1 = 0; p1 < P; ++p1)
                for (int n0 = 0; n0 < N0; ++n0)
                    for (int p0 = 0; p0 < P; ++p0)
                        pen << cbt[n0][n1].pos[0][p0][p1] << ',' << cbt[n0][n1].pos[1][p0][p1] << ',' << 0.0 << ','
                            << cbt[n0][n1].value[0][p0][p1] << ',' << cbt[n0][n1].value[1][p0][p1] << ',' << cbt[n0][n1].value[2][p0][p1] << ',' << cbt[n0][n1].value[3][p0][p1] << std::endl;
    }

    void vtkview(int interval)
    {
        static int timing = -1;
        ++timing;
        if (timing % interval)
            return;
        std::string file;
        if (timing > 99999)
            file = "view_99999.vtk";
        else if (timing > 9999)
            file = "view_" + std::to_string(timing) + ".vtk";
        else if (timing > 999)
            file = "view_0" + std::to_string(timing) + ".vtk";
        else if (timing > 99)
            file = "view_00" + std::to_string(timing) + ".vtk";
        else if (timing > 9)
            file = "view_000" + std::to_string(timing) + ".vtk";
        else
            file = "view_0000" + std::to_string(timing) + ".vtk";
        std::ofstream pen(file, std::ios::out | std::ios::trunc);
        pen << "# vtk DataFile Version 2.0" << std::endl
            << "The Penrose Steps are specious." << std::endl
            << "ASCII" << std::endl
            << "DATASET UNSTRUCTURED_GRID" << std::endl
            << "POINTS " << (P - 2) * (P - 2) * N0 * N1 << " float" << std::endl;
        for (int n0 = 0; n0 < N0; pen << std::endl, ++n0)
            for (int n1 = 0; n1 < N1; ++n1)
                for (int p1 = 1; p1 < P - 1; ++p1)
                    for (int p0 = 1; p0 < P - 1; ++p0)
                        pen << cbt[n0][n1].pos[0][p0][p1] << ' ' << cbt[n0][n1].pos[1][p0][p1] << ' ' << 0 << ' ';
        pen << "CELLS " << (P - 3) * (P - 3) * N0 * N1 << ' ' << (P - 3) * (P - 3) * N0 * N1 * 5 << std::endl;
        for (int n0 = 0; n0 < N0; ++n0)
            for (int n1 = 0; n1 < N1; ++n1)
                for (int p1 = 0; p1 < P - 3; ++p1)
                    for (int p0 = 0; p0 < P - 3; ++p0)
                        pen << 4 << ' ' << (n0 * N1 + n1) * (P - 2) * (P - 2) + p1 * (P - 2) + p0
                            << ' ' << (n0 * N1 + n1) * (P - 2) * (P - 2) + p1 * (P - 2) + p0 + 1
                            << ' ' << (n0 * N1 + n1) * (P - 2) * (P - 2) + (p1 + 1) * (P - 2) + p0 + 1
                            << ' ' << (n0 * N1 + n1) * (P - 2) * (P - 2) + (p1 + 1) * (P - 2) + p0 << std::endl;
        pen << "CELL_TYPES " << (P - 3) * (P - 3) * N0 * N1 << std::endl;
        for (int n0 = 0; n0 < (P - 3) * (P - 3) * N0 * N1; ++n0)
            pen << 9 << ' ';
        pen << std::endl
            << "POINT_DATA " << (P - 2) * (P - 2) * N0 * N1 << std::endl
            << "SCALARS values float 4" << std::endl
            << "LOOKUP_TABLE values" << std::endl;
        for (int n0 = 0; n0 < N0; pen << std::endl, ++n0)
            for (int n1 = 0; n1 < N1; ++n1)
                for (int p1 = 1; p1 < P - 1; ++p1)
                    for (int p0 = 1; p0 < P - 1; ++p0)
                        pen << cbt[n0][n1].value[0][p0][p1] << ' ' << cbt[n0][n1].value[1][p0][p1] << ' ' << cbt[n0][n1].value[2][p0][p1] << ' ' << cbt[n0][n1].value[3][p0][p1] << ' ';
    }
};

__global__ void slm(block *blk)
{
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        cubit &cbt = blk->cbt[blockIdx.x][blockIdx.y];
        cbt.idx[0] = &(blk->cbt[blockIdx.x][blockIdx.y + 1]), cbt.idx[1] = &(blk->cbt[blockIdx.x - 1][blockIdx.y]),
        cbt.idx[2] = &(blk->cbt[blockIdx.x][blockIdx.y - 1]), cbt.idx[3] = &(blk->cbt[blockIdx.x + 1][blockIdx.y]);
        if (blockIdx.x == 0)
            cbt.idx[1] = &(blk->cbt[N0 - 1][blockIdx.y]);
        if (blockIdx.x == N0 - 1)
            cbt.idx[3] = &(blk->cbt[0][blockIdx.y]);
        if (blockIdx.y == 0)
            cbt.idx[2] = &(blk->cbt[blockIdx.x][N1 - 1]);
        if (blockIdx.y == N1 - 1)
            cbt.idx[0] = &(blk->cbt[blockIdx.x][0]);
    }
}

__global__ void prs(block *blk, float radius = -1, int wall = 2, int far = 3)
{

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    {
        for (int n0 = 0; n0 < N0; ++n0)
            for (int n1 = 0; n1 < N1; ++n1)
                for (int i = 0; i < 4; ++i)
                    blk->cbt[n0][n1].idx[i] = nullptr;

        for (int n = 0; n < N0; ++n)
            blk->cbt[n][0].idx[0] = &(blk->cbt[n][2]), blk->cbt[n][0].idx[3] = &(blk->cbt[n][1]),
            blk->cbt[n][1].idx[0] = &(blk->cbt[n][0]), blk->cbt[n][1].idx[3] = &(blk->cbt[n][2]),
            blk->cbt[n][2].idx[0] = &(blk->cbt[n][1]), blk->cbt[n][2].idx[3] = &(blk->cbt[n][0]);

        for (int n = 0, t; n < N0; ++n)
            for (int i = 0; i < 3; ++i)
                if (blk->cbt[n][i].idx[2] == nullptr)
                    for (int m = n + 1; m < N0; ++m)
                        for (int j = 0; j < 3; ++j)
                            if (blk->cbt[m][j].idx[2] == nullptr && blk->compose[n][i] == blk->compose[m][j + 1] && blk->compose[n][i + 1] == blk->compose[m][j])
                                j + 1 == 3 ? t = 0 : t = j + 1, blk->cbt[n][i].idx[2] = &(blk->cbt[m][t]), blk->cbt[m][t].idx[1] = &(blk->cbt[n][i]),
                                             i + 1 == 3 ? t = 0 : t = i + 1, blk->cbt[n][t].idx[1] = &(blk->cbt[m][j]), blk->cbt[m][j].idx[2] = &(blk->cbt[n][t]);

        // for (int n = 0, t; n < N0; ++n)
        //     for (int i = 0; i < 3; ++i)
        //         if (blk->cbt[n][i].idx[2] == nullptr)
        //             for (int m = n + 1; m < N0; ++m)
        //                 for (int j = 0; j < 3; ++j)
        //                 {
        //                     if (blk->cbt[m][j].idx[2] == nullptr && blk->trinity[blk->compose[n][i] - 1][0] == blk->trinity[blk->compose[n][i + 1] - 1][0])
        //                         if (blk->trinity[blk->compose[n][i] - 1][1] == blk->trinity[blk->compose[m][j + 1] - 1][1] && blk->trinity[blk->compose[n][i + 1] - 1][1] == blk->trinity[blk->compose[m][j] - 1][1])
        //                             j + 1 == 3 ? t = 0 : t = j + 1, blk->cbt[n][i].idx[2] = &(blk->cbt[m][t]), blk->cbt[m][t].idx[1] = &(blk->cbt[n][i]),
        //                                          i + 1 == 3 ? t = 0 : t = i + 1, blk->cbt[n][t].idx[1] = &(blk->cbt[m][j]), blk->cbt[m][j].idx[2] = &(blk->cbt[n][t]);
        //                     if (blk->cbt[m][j].idx[2] == nullptr && blk->trinity[blk->compose[n][i] - 1][1] == blk->trinity[blk->compose[n][i + 1] - 1][1])
        //                         if (blk->trinity[blk->compose[n][i] - 1][0] == blk->trinity[blk->compose[m][j + 1] - 1][0] && blk->trinity[blk->compose[n][i + 1] - 1][0] == blk->trinity[blk->compose[m][j] - 1][0])
        //                             j + 1 == 3 ? t = 0 : t = j + 1, blk->cbt[n][i].idx[2] = &(blk->cbt[m][t]), blk->cbt[m][t].idx[1] = &(blk->cbt[n][i]),
        //                                          i + 1 == 3 ? t = 0 : t = i + 1, blk->cbt[n][t].idx[1] = &(blk->cbt[m][j]), blk->cbt[m][j].idx[2] = &(blk->cbt[n][t]);
        //                 }

        if (radius > 0.0)
            for (int n0 = 0; n0 < N0; ++n0)
                for (int n1 = 0; n1 < N1; ++n1)
                    for (int i = 0; i < 4; ++i)
                        if (blk->cbt[n0][n1].idx[i] == nullptr)
                        {
                            if (blk->cbt[n0][n1].q[0][0] * blk->cbt[n0][n1].q[0][0] + blk->cbt[n0][n1].q[0][1] * blk->cbt[n0][n1].q[0][1] < radius * radius)
                                blk->cbt[n0][n1].flg[i] = wall;
                            else
                                blk->cbt[n0][n1].flg[i] = far;
                        }
    };
}

__global__ void spect_to_value(int r, block *blk)
{
    cubit &cbt = blk->cbt[blockIdx.x][blockIdx.y];
    for (int d = 0; d < 4; ++d)
        cbt.value[d][threadIdx.x][threadIdx.y] = 0, cbt.value_x[d][threadIdx.x][threadIdx.y] = 0, cbt.value_y[d][threadIdx.x][threadIdx.y] = 0;
    for (int d = 0; d < 4; ++d)
        for (int s = 0; s < S; ++s)
            cbt.value[d][threadIdx.x][threadIdx.y] += cbt.spect[r][d][s] * cbt.base[s][threadIdx.x][threadIdx.y],
                cbt.value_x[d][threadIdx.x][threadIdx.y] += cbt.spect[r][d][s] * cbt.base_x[s][threadIdx.x][threadIdx.y],
                cbt.value_y[d][threadIdx.x][threadIdx.y] += cbt.spect[r][d][s] * cbt.base_y[s][threadIdx.x][threadIdx.y];
}

__global__ void value_to_spect(int r, block *blk)
{
    cubit &cbt = blk->cbt[blockIdx.x][blockIdx.y];
    if (threadIdx.x == 0 && threadIdx.y == 0)
        for (int d = 0; d < 4; ++d)
            for (int s = 0; s < S; ++s)
                cbt.spect[r][d][s] = 0;

    float cx[4], cy[4], vx[4], vy[4];
    cbt.convection(cx, cy, threadIdx.x, threadIdx.y), cbt.viscous(vx, vy, threadIdx.x, threadIdx.y);

    __shared__ float cx_s[4][P][P], cy_s[4][P][P], vx_s[4][P][P], vy_s[4][P][P];
    for (int d = 0; d < 4; ++d)
        cx_s[d][threadIdx.x][threadIdx.y] = cx[d], cy_s[d][threadIdx.x][threadIdx.y] = cy[d],
        vx_s[d][threadIdx.x][threadIdx.y] = vx[d], vy_s[d][threadIdx.x][threadIdx.y] = vy[d];
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
        for (int d = 0; d < 4; ++d)
            for (int s = 0; s < S; ++s)
                for (int p0 = 0; p0 < P; ++p0)
                    for (int p1 = 0; p1 < P; ++p1)
                        cbt.spect[r][d][s] += cx_s[d][p0][p1] * cbt.test_x[s][p0][p1], cbt.spect[r][d][s] += cy_s[d][p0][p1] * cbt.test_y[s][p0][p1],
                            cbt.spect[r][d][s] -= vx_s[d][p0][p1] * cbt.test_x[s][p0][p1], cbt.spect[r][d][s] -= vy_s[d][p0][p1] * cbt.test_y[s][p0][p1];
}

__global__ void rk(int r, float dt, block *blk)
{
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        cubit &cbt = blk->cbt[blockIdx.x][blockIdx.y];
        if (r == 1)
            for (int d = 0; d < 4; ++d)
                for (int s = 0; s < S; ++s)
                    cbt.spect[1][d][s] = cbt.spect[0][d][s] + cbt.spect[1][d][s] * dt;
        if (r == 2)
            for (int d = 0; d < 4; ++d)
                for (int s = 0; s < S; ++s)
                    cbt.spect[2][d][s] = (3.0 / 4.0) * cbt.spect[0][d][s] + (1.0 / 4.0) * cbt.spect[1][d][s] + (1.0 / 4.0) * cbt.spect[2][d][s] * dt;
        if (r == 3)
            for (int d = 0; d < 4; ++d)
                for (int s = 0; s < S; ++s)
                    cbt.spect[0][d][s] = (1.0 / 3.0) * cbt.spect[0][d][s] + (2.0 / 3.0) * cbt.spect[2][d][s] + (2.0 / 3.0) * cbt.spect[1][d][s] * dt;
    }
}

int main()
{
    dim3 B(N0, N1), G(P, P);
    block *blk_h = new block(), *blk_d;
    blk_h->bound(0, 3), blk_h->bound(2, 2), blk_h->slime("circlez.dat"), blk_h->vtkview(1);
    cudaMalloc((void **)&blk_d, sizeof(*blk_h)), cudaMemcpy(blk_d, blk_h, sizeof(*blk_h), cudaMemcpyHostToDevice);
    slm<<<B, G>>>(blk_d), cudaDeviceSynchronize();
    while (timer < end)
        timer = timer + dt, value_to_spect<<<B, G>>>(1, blk_d), cudaDeviceSynchronize(),
        rk<<<B, G>>>(1, dt, blk_d), cudaDeviceSynchronize(), spect_to_value<<<B, G>>>(1, blk_d), cudaDeviceSynchronize(), value_to_spect<<<B, G>>>(2, blk_d), cudaDeviceSynchronize(),
        rk<<<B, G>>>(2, dt, blk_d), cudaDeviceSynchronize(), spect_to_value<<<B, G>>>(2, blk_d), cudaDeviceSynchronize(), value_to_spect<<<B, G>>>(1, blk_d), cudaDeviceSynchronize(),
        rk<<<B, G>>>(3, dt, blk_d), cudaDeviceSynchronize(), spect_to_value<<<B, G>>>(0, blk_d), cudaMemcpy(blk_h, blk_d, sizeof(*blk_d), cudaMemcpyDeviceToHost),
        printf("shaladrassil: %f / %f\r", timer, end), blk_h->vtkview(1000);
}