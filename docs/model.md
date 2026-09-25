# Modelo formal: ACO multi-agente bidireccional para exploración heurística de contextos científicos

## 0. Alcance

Modelo de optimización para la **exploración heurística del espacio de citas** con el
fin de construir un **contexto óptimo** sobre una línea de investigación, partiendo de un
artículo semilla. Combina **Optimización por Colonia de Hormigas (ACO)** con **agentes LLM
persistentes**, **evaluación por comité triádico** (autoevaluación + voto de pares + juez
virgen + métricas estructurales) y **selección evolutiva** del mejor contexto. El grafo se
revela incrementalmente vía proveedores MCP.

---

## 1. Definición del problema

**Entradas:**
- Artículo semilla $s \in V$ (o conjunto $S \subseteq V$).
- Tema/consulta semilla $q$ (descripción textual de la línea).
- Presupuesto $B$ (cota: nº de fetches, profundidad, tiempo o convergencia).
- Proveedores MCP que exponen $\mathrm{Out}(u)$ (referencias) e $\mathrm{In}(u)$ (citantes) de
  un artículo $u$, así como sus metadatos.

**Salida:**
- La **narrativa** $N_{a^*}$ del agente $a^* = \arg\max_a Q_a$, donde $Q$ es una función de
  calidad del contexto construido.

**Objetivo:**

$$
a^* = \arg\max_{a \in \mathcal{A}} Q_a \quad \text{sujeto a} \quad \sum \text{fetches} \le B
$$

donde $\mathcal{A}$ es la colonia de agentes. El contexto es un subgrafo revelado
$G_{\text{exp}} \subseteq G$ junto con la narrativa sintetizada del agente ganador.

---

## 2. Definiciones preliminares

### 2.1 Grafo de citas

Sea $G = (V, E)$ un grafo dirigido donde $V$ es el conjunto de artículos y cada arista
$e = (a \to b) \in E$ denota que *$a$ cita a $b$*. Definimos:
- $\mathrm{Out}(u) = \{v \mid (u \to v) \in E\}$ — referencias de $u$.
- $\mathrm{In}(u) = \{v \mid (v \to u) \in E\}$ — citantes de $u$.
- $\mathcal{C}(u) = \mathrm{Out}(u) \cup \mathrm{In}(u)$ — vecindad total.

### 2.2 Modos de traversal

Una misma arista física $e = (a \to b)$ puede recorrerse en dos **modos** desde el nodo
actual $u$:

| Modo $d$ | Significado | Movimiento | Dirección temporal |
|---|---|---|---|
| `ref` | seguir una referencia | $u \to v$, con $v \in \mathrm{Out}(u)$ | hacia atrás (fundamentos) |
| `cites` | seguir un citante | $u \to v$, con $v \in \mathrm{In}(u)$ | hacia adelante (impacto) |

Se mantienen **dos matrices de feromona independientes**: $\tau_{\text{ref}},
\tau_{\text{cites}} : E \to \mathbb{R}_+$.

### 2.3 Estado del agente

Cada agente $a \in \mathcal{A}$ mantiene (estado persistible entre turnos):

$$
\sigma_a = \big(\, \text{pos}_a,\; V_a,\; N_a,\; b_a,\; Q_a,\; F_a \,\big)
$$

- $\text{pos}_a \in V$ — posición actual.
- $V_a \subseteq V$ — artículos visitados (fetcheados).
- $N_a$ — narrativa acumulada (texto, síntesis evolutiva).
- $b_a \in \mathbb{N}$ — presupuesto restante.
- $Q_a \in [0,1]$ — calidad actual.
- $F_a = \left(\bigcup_{v \in V_a} \mathcal{C}(v)\right) \setminus V_a$ — frontera (candidatos
  revelados no visitados).

### 2.4 Estado global (compartido)

$$
\Sigma = \big(\, \tau_{\text{ref}},\; \tau_{\text{cites}},\; M,\; G_{\text{exp}},\; \eta \,\big)
$$

- $\tau_{\text{ref}}, \tau_{\text{cites}}$ — feromonas por modo.
- $M : V \to \text{metadatos}$ — caché (título, abstract, año, citas, embedding).
- $G_{\text{exp}}$ — subgrafo revelado.
- $\eta : V \to [0,1]$ — caché de relevancia (juez virgen), **una vez por nodo, compartida
  por toda la colonia**.
- $\Gamma$ — conjunto de **reclamaciones** (claims) sobre la frontera: nodos que un agente
  está fetcheando en ese momento (bloqueo solo de la ventana concurrente).

> **Propiedad de persistencia ligera:** como $\eta$ y $\tau$ son globales, el estado
> serializado por agente se reduce a IDs + texto narrativo + escalares, lo que hace viable
> la multitarea bajo una cota estricta de concurrencia $K$.

---

## 3. Heurística local $\eta(v)$ — componentes normalizados

Estima la relevancia de un candidato $v$ **antes** de fetchearlo a fondo, usando
metadatos disponibles en la lista de referencias del padre. Se descompone en
**componentes normalizados** cada uno en $[0,1]$:

$$
\eta(v) = \sigma\!\left( w_s \cdot \mathrm{sim}(v) \;+\; w_c \cdot \mathrm{cit}(v)
\;+\; w_y \cdot \mathrm{rec}(v) \;+\; w_{conf} \cdot \mathrm{conf}(v)
\;+\; w_{llm} \cdot \mathrm{llm}(v) \right)
$$

con $\sigma$ la sigmoide y:
- $\mathrm{sim}(v)$ — similitud coseno normalizada $(\cos+1)/2$ entre el embedding
  (título/abstract corto) de $v$ y la consulta semilla $q$.
- $\mathrm{cit}(v) \in [0,1]$ — citas normalizadas $\sigma(\ln(1+\text{citas})/5)$
  (señal de seminalidad).
- $\mathrm{rec}(v) \in [0,1]$ — factor de año $(y-1950)/75$ recortado.
- $\mathrm{conf}(v) \in [0,1]$ — confianza de proveedor/verificación: base
  determinista por proveedor (openalex 0.95, semantic_scholar 0.90, pubmed 0.85,
  arxiv 0.80, resto 0.40) ajustada por evidencia (DOI/arXiv, abstract,
  ausencia de título).
- $\mathrm{llm}(v)$ — prioridad LLM opcional (solo se incluye cuando `eta_llm = true`;
  la puntuación del lote de evaluación de referencias). Contribuye con peso $w_{llm}$.

**Optimización de coste:** por defecto $\eta$ es **pura** (sin término LLM), pues
se evalúa para cada candidato de cada agente. El **LLM se usa opcionalmente como
prioridad** cuando se activa `eta_llm` (toggle del config). El modo $\eta$ es
siempre cacheado globalmente.

**Caché:** $\eta(v)$ se calcula una vez por nodo y se comparte globalmente; sus
componentes se registran en el evento de replay `candidate_score`.

---

## 4. Probabilidad de transición bidireccional

Se compone de: **peso de dirección** (ajustado por casta) y **peso de arista**
$\tau^\alpha \cdot \eta^\beta$.

### 4.1 Peso de dirección por modo

$$
p(d \mid u) = \frac{w_d^{\text{casta}}(u)}{w_{\text{ref}}^{\text{casta}}(u) + w_{\text{cites}}^{\text{casta}}(u)}, \qquad d \in \{\text{ref}, \text{cites}\}
$$

donde $w_d^{\text{casta}}$ es la línea base del config ($\theta$), ajustada por casta:
- **fundaciones**: desplaza el peso hacia `ref` (fundamentos).
- **impacto**: desplaza el peso hacia `cites` (impacto).
- **mixto**: usa la línea base del config (equilibrada).

Los pesos se normalizan a suma 1; el modificador de dirección de un candidato se
aplica como factor en su peso (§4.2).

### 4.2 Peso de arista dentro del modo

$$
\mathrm{weight}(v) = \tau^{\alpha}_{\text{modo}}(u,v) \cdot \eta(v)^{\beta} \cdot d^{\text{casta}}_{\text{modo}(v)}
$$

y la probabilidad de selección se normaliza sobre la frontera elegible:

$$
p(v) = \frac{\mathrm{weight}(v)}{\sum_{w \in \mathcal{C} \setminus V_a^{\text{claim}}} \mathrm{weight}(w)}
$$

- $\tau_{\text{modo}}(u,v)$ — **feromona privada** del agente en la arista
  $(u \to v)$ segun el modo de descubrimiento, depositada por él y posteriormente
  usada **por él** (los agentes no comparten feromona).
- $V_a^{\text{claim}}$ — nodos visitados **o reclamados** por otro agente.

### 4.3 Exploración $\varepsilon$-greedy e RNG inyectable

Con probabilidad $\varepsilon$, se elige un candidato **uniformemente** sobre la
frontera elegible (exploración pura, ignorando $\tau$ y $\eta$). El RNG es
**inyectable y determinista** (semilla por agente en la colonia), de modo que una
ejecución con la misma semilla es reproducible.

### 4.4 Reclamación (claims) sin borrado global

La frontera es compartida por la colonia. Para evitar que dos agentes fetcheen el
mismo candidato a la vez, el agente **reclama** (claim) el nodo durante el fetch y
lo **libera** al terminar. Reclamar solo bloquea la ventana concurrente: no borra
el nodo ni los caminos ya descubiertos que cuelgan de él, de modo que las rutas
reutilizables se conservan.

---

## 5. Función de calidad $Q$ — comité triádico

Se evalúa **al final de cada turno** (tras $k$ artículos fetcheados), sobre el delta
$\Delta_a(t)$ y el contexto acumulado:

$$
Q_a = w_1\, S_a + w_2\, P_a + w_3\, J_a + w_4\, R_a, \qquad \sum_{i=1}^{4} w_i = 1
$$

| Símbolo | Componente | Fuente | LLM calls |
|---|---|---|---|
| $S_a$ | Autoevaluación de la narrativa $N_a$ | el propio agente | $1$ |
| $P_a$ | Voto de pares sobre $\Delta_a(t)$ | otros $K{-}1$ agentes activos | $K{-}1$ |
| $J_a$ | Juez virgen (imparcial, sin contexto acumulado) | LLM fresco | $1$ |
| $R_a$ | Métricas estructurales (determinista) | cálculo | $0$ |

### 5.1 Voto de pares agregado

**Mediana (robusta):**

$$
P_a = \operatorname{median}_{a' \neq a}\, P_{a' \to a}
$$

**Ponderada por reputación (opcional):** los agentes de mayor $Q$ pesan más:

$$
P_a = \frac{\sum_{a' \neq a} Q_{a'} \cdot P_{a' \to a}}{\sum_{a' \neq a} Q_{a'}}
$$

### 5.2 Métricas estructurales $R_a$

$$
R_a = \mu_1\, \text{Cobertura}_a + \mu_2\, \text{Diversidad}_a + \mu_3\,
\text{Profundidad}_a + \mu_4\, \text{Coherencia}_a, \quad \sum \mu_i = 1
$$

- **Cobertura:** $\mathrm{Cob}_a = \min\!\left(1,\; |V_a|/L\right)$ con $L$ un tamaño de
  contexto objetivo.
- **Diversidad:** $\mathrm{Div}_a = 1 - \dfrac{2}{|V_a|(|V_a|-1)} \sum_{v \neq w \in V_a}
  \mathrm{sim}_{\text{emb}}(v,w)$ — penaliza la redundancia semántica.
- **Profundidad/seminalidad:** $\mathrm{Prof}_a = \dfrac{1}{|V_a|}\sum_{v \in V_a}
  \mathrm{normCitas}(v)$.
- **Coherencia:** $\mathrm{Coh}_a$ = fracción de $V_a$ que pertenece a un linaje conectado
  desde $s$ — penaliza conjuntos dispersos.

### 5.3 Señal de mejora

$$
\Delta Q_a(t) = Q_a(t) - Q_a(t-1)
$$

$\Delta Q_a(t) > 0$ indica que el turno aportó valor; impulsa el depósito de feromona (§6)
y la prioridad del scheduler (§8).

---

## 6. Actualización de feromona

Tras cada oleada (todos los agentes activos completaron su turno):

### 6.1 Evaporación

$$
\tau_d(e) \leftarrow (1 - \rho)\, \tau_d(e), \qquad \forall e,\; \forall d
$$

### 6.2 Depósito por agente

$$
\tau_d(e) \leftarrow \tau_d(e) + \sum_{a} \frac{\Delta Q_a(t)^{+}}
{|\text{path}_a(t)|}\, \mathbb{1}\!\left[e \in \text{path}_a(t)\right]
$$

con $\Delta Q_a(t)^{+} = \max(0, \Delta Q_a(t))$ y normalización por longitud de camino
para no sesgar hacia caminos cortos.

### 6.3 Elitismo

$$
\tau_d(e) \leftarrow \tau_d(e) + \lambda\, Q_{\text{best}}\, \mathbb{1}\!\left[e \in
\text{best\_path}\right]
$$

### 6.4 Cotas Max-Min (MMAS)

$$
\tau_d(e) \leftarrow \operatorname{clip}\!\left(\tau_d(e),\; \tau_{\min},\;
\tau_{\max}\right)
$$

$\tau_{\min}$ garantiza exploración residual; $\tau_{\max}$ evita monopolio de rutas.

---

## 7. Narrativa como representación del contexto

$N_a$ es un **documento vivo** actualizado tras cada fetcheo mediante un operador de
integración:

$$
N_a \leftarrow \mathrm{Integrate}_{\text{LLM}}\!\left(N_a,\; v\right)
$$

*"Integra la contribución de $v$ en tu comprensión actual de la línea de investigación."*

Cumple triple rol:
1. **Salida** — la narrativa del agente $\arg\max_a Q_a$ es el entregable.
2. **Sustrato de evaluación** — pares y juez virgen leen $N_a$ (o un resumen) para votar.
3. **Contexto de razonamiento** — soporta la autoevaluación $S_a$.

> La narrativa captura *relaciones y comprensión*, no un saco de artículos — que es lo que
> realmente significa "contexto de la línea".

---

## 8. Planificador de concurrencia

Con cota de $K$ agentes activos simultáneos, la colonia $|\mathcal{A}| = N$ se segmenta en
turnos. Prioridad de un agente dormido:

$$
\text{priority}(a) = Q_a \;+\; \gamma \cdot \mathrm{EV}(F_a) \;+\; \delta \cdot b_a
$$

- $\mathrm{EV}(F_a) = \sum_{v \in F_a} p(\text{pos}_a \to v)\, \eta(v)$ — valor esperado de
  la frontera (nodos reclamados excluidos).
- $b_a$ — recompensa el presupuesto restante (evita inanición).

El scheduler extrae los $K$ agentes de mayor prioridad, los ejecuta un turno, los persiste,
recalcula prioridades y repite. Es **multitarea cooperativa con prioridad** que concentra
el cómputo en los linajes más prometedores. $K$ es un parámetro (el límite de 3 es un
perfil).

El recuento de trabajo de proveedor es **consistente**: cada unidad de presupuesto consumida
se contabiliza como trabajo de fetch, incluidos los **tránsitos de nodo solo-metadatos** y
la expansión de vecinos del proveedor ($\Delta_{\text{turno}} = \text{visitas} +
\text{tránsitos}$ por agente), de modo que el presupuesto y la convergencia no subestiman
el trabajo de expansión.

---

## 9. Criterios de convergencia

Parada cuando **cualquiera** (configurable):
- **Presupuesto:** $\sum_a b_a = 0$ o $\sum \text{fetches} = B$ (incluye trabajo de
  expansión de proveedor y tránsitos de nodos solo-metadatos).
- **Meseta:** $Q_{\text{best}}(t) - Q_{\text{best}}(t-T) < \varepsilon$ durante $T$ oleadas.
- **Concentración de feromona:** $\max_e \tau(e) / \overline{\tau(e)} > \theta$.
- **Tiempo** wall-clock.

Los nodos de frontera **solo-metadatos** (resolubles vía expansión del proveedor, sin
contenido legible) cuentan como candidatos y su expansión se contabiliza como trabajo de
proveedor, de modo que una colonia atascada en metadatos agota su presupuesto y converge
en lugar de girar indefinidamente sin avanzar.

Al converger: $a^* = \arg\max_a Q_a$; emitir $N_{a^*}$.

---

## 10. Algoritmo

```
Entrada: semilla s, consulta q, presupuesto B, parámetros (N, K, k, α, β, ρ, ε, w, λ,
         τ_min, τ_max, pesos η, pesos de dirección)
Salida:  narrativa N_{a*}

1.  Inicializar colonia A = {a_1..a_N} en s, N_a = ∅, V_a = {s}, b_a = B/N
2.  Revelar C(s) vía proveedor; G_exp = {s}; τ_d(e) = τ_0 ∀e,d; η(s) calculado
3.  Mientras no converja (§9):
4.      A_activos ← top-K agentes por priority(a) (§8)
5.      Para cada a ∈ A_activos (turno):
6.          Para i = 1..k y b_a > 0:
7.              Frontera elegible ← C_u \ (V_a ∪ claims)
8.              Con prob. ε: v ~ Uniforme(elegibles)                (§4.3)
9.              Si no:   v ~ p(v) ∝ τ^α · η^β · d_castaa         (§4)
10.             reclamar v (claim); bajar/soltar al terminar         (§4.4)
11.             Fetchear v vía proveedor (si no en M); actualizar M, G_exp, η(v)
12.             Si v es solo-metadatos: expandir y continuar (sin crédito)
13.             V_a ← V_a ∪ {v}; pos_a ← v; b_a ← b_a - 1
14.             N_a ← Integrate_LLM(N_a, v)              (§7)
15.         Fin-para
16.         Calcular S_a, P_a (voto de otros activos), J_a (virgen), R_a  (§5)
17.         Q_a ← Σ w_i · componente_i ;  ΔQ_a ← Q_a - Q_a_previo
18.         Persistir σ_a; liberar slot
19.     Fin-para
20.     Actualizar τ: evaporar, depositar por ΔQ, elitismo, clip [τ_min,τ_max]  (§6)
21.     Recalcular priority(a) ∀a dormido
22. Fin-mientras
23. a* ← argmax_a Q_a ;  devolver N_{a*}
```

---

## 11. Parámetros

| Símbolo | Descripción | Típico |
|---|---|---|
| $N$ | Tamaño de la colonia | 10–30 |
| $K$ | Concurrencia (slots activos) | 1–5 (caso personal: 3) |
| $k$ | Artículos por turno | 2–6 |
| $B$ | Presupuesto global | configurable |
| $\alpha$ | Peso feromona | 1 |
| $\beta$ | Peso heurística $\eta$ | 2–5 |
| $\rho$ | Evaporación | 0.1–0.5 |
| $\varepsilon$ | Exploración $\varepsilon$-greedy | 0.05–0.2 |
| $w_1..w_4$ | Pesos de $Q$ (S,P,J,R) | p.ej. 0.25 |
| $\mu_1..\mu_4$ | Pesos de $R$ | p.ej. 0.25 |
| $\theta_{\text{ref}}, \theta_{\text{cites}}$ | Pesos de dirección | 0.7 / 0.3 |
| $w_s, w_c, w_y, w_{conf}, w_{llm}$ | Pesos de $\eta$ (sim, citas, recencia, confianza, LLM) | 0.6 / 0.2 / 0.2 / 0.1 / 0.2 |
| $\text{eta\_llm}$ | Usar prioridad LLM en $\eta$ | false |
| $\lambda$ | Elitismo | 0.5–1 |
| $\tau_0, \tau_{\min}, \tau_{\max}$ | Cotas MMAS (inicial, mín, máx) | 1, 0.1, 10 |
| $\gamma, \delta$ | Scheduler (EV, presupuesto) | 0.5, 0.3 |

---

## 12. Complejidad y coste por oleada

Con $K$ activos y $k$ fetches/turno:

| Concepto | LLM calls | Cacheable |
|---|---|---|
| Integración narrativa | $k \cdot K$ | No |
| Autoevaluación $S$ | $K$ | No |
| Voto de pares $P$ | $K(K-1)$ | No |
| Juez virgen $J$ | $K$ | No |
| Estructural $R$ | $0$ | — |
| $\eta(v)$ | $0$ (embedding) | **Sí, global** |

Total $\approx K^2 + (k+2)K$ LLM calls/oleada. Para $K=3, k=4$: $\approx 24$ calls/oleada.
Escala linealmente con $K$ y $k$.

---

## 13. Extensiones

- **Castas:** asignar a cada agente una casta $c \in \{\text{fundaciones}, \text{impacto},
  \text{mixto}\}$ que fija los pesos de dirección $w_d^{\text{casta}}$ (§4.1):
  `fundaciones` desplaza hacia `ref`, `impacto` hacia `cites`, `mixto` usa la línea base
  equilibrada. La colonia cubre ambas direcciones por especialización; el ganador emerge de
  la casta que mejor capturó el contexto. Aumenta la diversidad (término $\mathrm{Div}_a$)
  sin coste extra.
- **Reclamación (claims) y feromona privada:** los agentes reclaman nodos de la frontera
  compartida para evitar trabajo duplicado concurrente (§4.4) y se refuerzan sobre su propia
  feromona privada, sin borrar rutas reutilizables.
- **Replay de candidatos:** cada candidato puntuado emite `candidate_score` (todos los
  componentes/ponderaciones y $\eta$) y cada selección `candidate_selected` ($\tau$,
  $\alpha/\beta$, modificador de dirección/casta, peso y probabilidad finales, rama
  $\varepsilon$ y elección), dotando a la traza de la regla exacta que escogió cada salto.
- **Voto ponderado por reputación:** §5.1, los agentes de mayor $Q$ pesan más en $P_a$.
- **Reanudación:** $\Sigma$ y $\{\sigma_a\}$ son serializables → ejecuciones
  pausables/reanudables.

---

## 14. Relación con la literatura ACO

- **Sistema de Hormigas (AS)** — transición $\tau^\alpha \eta^\beta$ (Dorigo, 1992).
- **Max-Min Ant System (MMAS)** — cotas $[\tau_{\min}, \tau_{\max}]$ (Stützle & Hoos, 2000).
- **ACO elitista** — depósito extra del mejor agente (§6.3).
- **Multi-colonia / castas** — subpoblaciones especializadas (§13).
- **Novedad:** hormigas **inteligentes y persistentes** con narrativa, evaluación por
  **comité triádico** (auto + pares + virgen) y **selección evolutiva** del mejor contexto;
  el grafo se revela **online** vía MCP.

---

## 15. Glosario de notación

| Símbolo | Significado |
|---|---|
| $G=(V,E)$ | Grafo de citas dirigido |
| $s, q$ | Artículo semilla / consulta semilla |
| $\mathrm{Out}(u), \mathrm{In}(u)$ | Referencias / citantes de $u$ |
| $\mathcal{C}(u)$ | Vecindad total de $u$ |
| $d \in \{\text{ref},\text{cites}\}$ | Modo de traversal |
| $\tau_d(e)$ | Feromona de la arista $e$ en modo $d$ |
| $\eta(v)$ | Heurística local (juez virgen, cacheable) |
| $\sigma_a$ | Estado del agente $a$ |
| $V_a, N_a, F_a, b_a, Q_a$ | Visitados, narrativa, frontera, presupuesto, calidad |
| $S_a, P_a, J_a, R_a$ | Autoevaluación, pares, virgen, estructural |
| $\Delta Q_a(t)$ | Señal de mejora por turno |
| $K, k, N$ | Concurrencia, fetches/turno, tamaño colonia |
| $\alpha,\beta,\rho,\varepsilon$ | Feromona, heurística, evaporación, exploración |
| $\Gamma$ | Reclamaciones (claims) sobre la frontera compartida |
| $w_{conf}, w_{llm}, \text{eta\_llm}$ | Confianza de proveedor y prioridad LLM en $\eta$ |
