// 384-bit reversible tridiagonal propagator for stiff Bayesian proposals.
// Independent implicit-QL diagonalization with accumulated plane rotations.
#ifdef HUMMER_QUAD
#include <quadmath.h>
#else
#include <mpfr.h>
#endif
#include <vector>
#include <limits>
#include <cmath>
#ifdef HUMMER_QUAD
using Real=__float128;
Real decimal(const char*s) {return strtoflt128(s,nullptr);}
Real abs(Real a) {return fabsq(a);}
Real sqrt(Real a) {return sqrtq(a);}
Real exp(Real a) {return expq(a);}
Real log(Real a) {return logq(a);}
#else
class Real {
 public:
  mpfr_t v;
  Real() { mpfr_init2(v,384);mpfr_set_ui(v,0,MPFR_RNDN); }
  Real(int a):Real() { mpfr_set_si(v,a,MPFR_RNDN); }
  Real(double a):Real() { mpfr_set_d(v,a,MPFR_RNDN); }
  Real(const char* a):Real() { mpfr_set_str(v,a,10,MPFR_RNDN); }
  Real(const Real& a):Real() { mpfr_set(v,a.v,MPFR_RNDN); }
  ~Real() { mpfr_clear(v); }
  Real& operator=(const Real& a) { if(this!=&a)mpfr_set(v,a.v,MPFR_RNDN);return *this; }
  Real& operator+=(const Real& a) { mpfr_add(v,v,a.v,MPFR_RNDN);return *this; }
  Real& operator-=(const Real& a) { mpfr_sub(v,v,a.v,MPFR_RNDN);return *this; }
  Real& operator*=(const Real& a) { mpfr_mul(v,v,a.v,MPFR_RNDN);return *this; }
  Real& operator/=(const Real& a) { mpfr_div(v,v,a.v,MPFR_RNDN);return *this; }
  explicit operator double() const { return mpfr_get_d(v,MPFR_RNDN); }
};
Real operator+(Real a,const Real& b) {return a+=b;}
Real operator-(Real a,const Real& b) {return a-=b;}
Real operator*(Real a,const Real& b) {return a*=b;}
Real operator/(Real a,const Real& b) {return a/=b;}
Real operator-(const Real& a) {Real b;mpfr_neg(b.v,a.v,MPFR_RNDN);return b;}
bool operator<(const Real&a,const Real&b) {return mpfr_cmp(a.v,b.v)<0;}
bool operator>(const Real&a,const Real&b) {return mpfr_cmp(a.v,b.v)>0;}
bool operator<=(const Real&a,const Real&b) {return mpfr_cmp(a.v,b.v)<=0;}
bool operator>=(const Real&a,const Real&b) {return mpfr_cmp(a.v,b.v)>=0;}
bool operator==(const Real&a,const Real&b) {return mpfr_cmp(a.v,b.v)==0;}
Real abs(const Real&a) {Real b;mpfr_abs(b.v,a.v,MPFR_RNDN);return b;}
Real sqrt(const Real&a) {Real b;mpfr_sqrt(b.v,a.v,MPFR_RNDN);return b;}
Real exp(const Real&a) {Real b;mpfr_exp(b.v,a.v,MPFR_RNDN);return b;}
Real log(const Real&a) {Real b;mpfr_log(b.v,a.v,MPFR_RNDN);return b;}
Real decimal(const char*s) {return Real(s);}
#endif

extern "C" int hummer_loglike_mp(int n, const double* free_g, const double* rates,
                                const double* counts, double tau,
                                double* loglike, double* transition) {
  try {
    if(n<2 || n>128 || !(tau>0)) return 1;
    std::vector<Real> p(n), d(n), e(n), z(n*n,Real(0));
    Real norm=0;
    for(int i=0;i<n;++i) {
      p[i]=exp(i==n-1 ? Real(0) : -Real(free_g[i])); norm+=p[i];
      z[i*n+i]=1;
    }
    for(auto &v:p) v/=norm;
    for(int i=0;i<n-1;++i) {
      if(!(rates[i]>0) || !std::isfinite(rates[i])) return 2;
      Real k=Real(rates[i]), back=k*p[i]/p[i+1];
      d[i]-=k; d[i+1]-=back; e[i]=k*sqrt(p[i]/p[i+1]);
    }
    e[n-1]=0;
#ifdef HUMMER_QUAD
    const Real eps=decimal("1e-33");
#else
    const Real eps=decimal("1e-95");
#endif
    for(int l=0;l<n;++l) {
      int iterations=0;
      while(true) {
        int m=l;
        for(;m<n-1;++m) {
          if(abs(e[m])<=eps*(abs(d[m])+abs(d[m+1])+Real(1))) break;
        }
        if(m==l) break;
        if(++iterations>300) return 3;
        Real g=(d[l+1]-d[l])/(Real(2)*e[l]);
        Real r=sqrt(g*g+1);
        g=d[m]-d[l]+e[l]/(g+(g>=0?r:-r));
        Real s=1,c=1,shift=0;
        for(int i=m-1;i>=l;--i) {
          Real f=s*e[i], b=c*e[i];
          if(abs(f)>=abs(g)) {
            if(f==0) { c=1;s=0;e[i+1]=0; }
            else { c=g/f;r=sqrt(c*c+1);e[i+1]=f*r;s=1/r;c*=s; }
          } else {
            s=f/g;r=sqrt(s*s+1);e[i+1]=g*r;c=1/r;s*=c;
          }
          g=d[i+1]-shift;
          r=(d[i]-g)*s+Real(2)*c*b;
          shift=s*r;d[i+1]=g+shift;g=c*r-b;
          for(int row=0;row<n;++row) {
            f=z[row*n+i+1];
            z[row*n+i+1]=s*z[row*n+i]+c*f;
            z[row*n+i]=c*z[row*n+i]-s*f;
          }
        }
        d[l]-=shift;e[l]=g;e[m]=0;
      }
    }
    // The zero eigenvalue is known analytically, avoiding amplification of its
    // tiny rounding residual when the other eigenvalues span many decades.
    int stationary=0;
    for(int i=1;i<n;++i) if(abs(d[i])<abs(d[stationary])) stationary=i;
    d[stationary]=0;
    std::vector<Real> weights(n);
    for(int i=0;i<n;++i) {
      if(d[i]>decimal("1e-25")) return 4;
      if(d[i]>0) d[i]=0;
      weights[i]=exp(Real(tau)*d[i]);
    }
    Real ll=0;
    std::vector<Real> colsum(n,Real(0));
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
      Real value=0;
      for(int k=0;k<n;++k) value+=z[i*n+k]*weights[k]*z[j*n+k];
      value*=sqrt(p[i]/p[j]);
      if(value<decimal("-1e-25") || value>Real(1)+decimal("1e-25")) return 5;
      colsum[j]+=value;
      if(counts[i*n+j]>0) {
        if(value<=0) return 6;
        ll+=Real(counts[i*n+j])*log(value);
      }
      if(transition) transition[i*n+j]=static_cast<double>(value);
    }
    for(auto v:colsum) if(abs(v-1)>decimal("1e-20")) return 7;
    *loglike=static_cast<double>(ll);
    return std::isfinite(*loglike) && *loglike<=1e-8 ? 0 : 8;
  } catch(...) { return 9; }
}
