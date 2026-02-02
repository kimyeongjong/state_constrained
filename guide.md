
# 고차원 원기둥 도메인 (Generalized k Version)  
(Inequality, Uniform Sampling, Distance Function)

본 문서는 다음 두 simplex

$$
\sum_{i=1}^n x_i = +k,\quad x_i \ge 0
$$
$$
\sum_{i=1}^n x_i = -k,\quad x_i \le 0
$$

에서 만들어지는 **확대된 원기둥 도메인 $\Omega_k$** 에 대해

- 부등식 형태
- uniform sampling
- 거리 함수 $d(x,\Omega_k)$
- PDE

를 정리한 파일이다.

---

# 1. 두 simplex의 inball

## 1.1 incenter

양쪽 simplex는 모두 정단체이므로 incenter는

$$
c_+ = \left(\frac{k}{n},\dots,\frac{k}{n}\right),\qquad
c_- = -c_+.
$$

## 1.2 inradius

원래 단위(simplex sum=1) 라면 inradius = $1/\sqrt{n(n-1)}$.  
sum=±k 로 확대되었으므로 길이비가 k배 커짐:

$$
r_k = k\,\frac{1}{\sqrt{n(n-1)}}.
$$

---

# 2. 도메인 \(\Omega_k\) 의 부등식

다음 두 함수를 정의:

$$
S(x)=\sum_{i=1}^n x_i,\qquad
Q(x)=\sum_{i=1}^n x_i^2 - \frac{1}{n}\left(\sum_{i=1}^n x_i\right)^2.
$$

그러면 도메인 $\Omega_k$ 는

$$
\boxed{
\Omega_k
=
\left\{
x\in\mathbb{R}^n :
\ |S(x)|\le k,\quad
Q(x)\le r_k^2 = \frac{k^2}{n(n-1)}
\right\}.
}
$$

---

# 3. 구조적 해석

축 방향 단위벡터:

$$
u_0=\frac{1}{\sqrt{n}}(1,\dots,1),
$$

축 방향 좌표:

$$
a = \frac{S(x)}{\sqrt{n}}.
$$

단면 성분:

$$
\|v\|^2 = Q(x).
$$

높이 제한은 $|S(x)| \le k$, 즉

$$
|a| \le \frac{k}{\sqrt{n}}.
$$

단면 제한은

$$
\|v\| \le r_k = \frac{k}{\sqrt{n(n-1)}}.
$$

따라서

$$
\boxed{
\Omega_k \cong [-k,k] \times B_{r_k}^{\,n-1}.
}
$$

---

# 4. Uniform Sampling (확대된 k 버전)

## Step 1: 축 방향

$$
s \sim \mathrm{Uniform}([-k,k]).
$$

## Step 2: 단면 샘플링

1. $w\sim N(0, I_n)$
2. $\bar w=\frac{1}{n}\sum w_i$
3. $v = w-\bar w (1,\dots,1)$
4. $\hat v = v/\|v\|$
5. $U\sim \mathrm{Uniform}(0,1)$
6. $\rho = r_k\,U^{1/(n-1)}$
7. $u=\rho \hat v$

## Step 3: 최종 샘플

$$
x = \left(\frac{s}{n},\dots,\frac{s}{n}\right)+u.
$$

---

# 5. Uniform Sampling 의사코드

```pseudo
Input: n, k
r_k = k / sqrt(n*(n-1))

1. sample s ~ Uniform(-k, k)

2. sample w ~ N(0, I_n)
   mean_w = (1/n)*sum(w)
   v = w - mean_w*(1,...,1)
   v_hat = v / ||v||

3. sample U ~ Uniform(0,1)
   rho = r_k * U^(1/(n-1))
   u = rho * v_hat

4. m = (s/n,...,s/n)
5. x = m + u
return x
```

---

# 6. 거리 함수 \(d(x,\Omega_k)\)

축 방향 범위:

$$
|a| \le \frac{k}{\sqrt{n}}.
$$

단면 범위:

$$
\|v\|\le r_k=\frac{k}{\sqrt{n(n-1)}}.
$$

## 6.1 축 방향 거리

$$
d_\parallel=
\begin{cases}
0, & |a|\le k/\sqrt{n},\\
|a|-k/\sqrt{n}, & |a|>k/\sqrt{n}.
\end{cases}
$$

## 6.2 단면 거리

$$
d_\perp=
\begin{cases}
0, & \|v\|\le r_k,\\
\|v\|-r_k, & \|v\|>r_k.
\end{cases}
$$

## 6.3 최종 거리

$$
\boxed{
d(x,\Omega_k)=\sqrt{d_\parallel^2 + d_\perp^2}.
}
$$

---

# 6.4 거리 계산 의사코드

```pseudo
Input: x, n, k
s = sum(x)
norm2 = sum(x_i^2)
a = s / sqrt(n)
v_norm = sqrt(max(norm2 - a*a, 0))

a_max = k / sqrt(n)
r_k = k / sqrt(n*(n-1))

if abs(a) <= a_max:
    d_parallel = 0
else:
    d_parallel = abs(a) - a_max

if v_norm <= r_k:
    d_perp = 0
else:
    d_perp = v_norm - r_k

return sqrt(d_parallel^2 + d_perp^2)
```

---

# 7. Hamiltonian-Jacobi equation

$$
H(p)=\frac{\|p-\mathbf{1}\|_1}{n} - 1,
$$
where $p\in [0,2]^n$ and $\mathbf{1}$ is a vector with all of its component equal to $1$.

The Hamilton-Jacobi equation that we aim to solve on $\Omega_k$ is
$$
u(x)+H(\nabla u)=0.
$$

$u=\exp(x_1+x_2+\ldots+x_n-k)$ is the unique viscosity solution of this equation. We want to approximate $u$ by iteratively solve the equation
$$
u(x)+H(\nabla u)=d(x,\Omega_k)/\epsilon
$$
using PINN as $\epsilon$ goes to zero.

- The viscosity solution: $u_k(x)$
