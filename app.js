document.addEventListener('DOMContentLoaded', () => {
    let datosGlobales = null;

    // Elementos DOM
    const btnRecalcular = document.getElementById('btn-recalcular');
    const statusBanner = document.getElementById('status-banner');
    const statusText = document.getElementById('status-text');

    const sliderVanCap = document.getElementById('slider-van-cap');
    const vanCapVal = document.getElementById('van-cap-val');

    const inputBuscar = document.getElementById('input-buscar-estacion');
    const selectFiltro = document.getElementById('select-filtro-alerta');

    // Inicializar Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Escuchar Slider de la Furgoneta
    sliderVanCap.addEventListener('input', (e) => {
        vanCapVal.textContent = e.target.value;
    });

    sliderVanCap.addEventListener('change', () => {
        cargarDatos(sliderVanCap.value);
    });

    // Botón Recalcular
    btnRecalcular.addEventListener('click', () => {
        cargarDatos(sliderVanCap.value);
    });

    // Filtros de búsqueda
    inputBuscar.addEventListener('input', renderTablaEstaciones);
    selectFiltro.addEventListener('change', renderTablaEstaciones);

    // Función principal para consultar la API Serverless
    async function cargarDatos(capacidadVan = 10) {
        setLoadingState(true);

        try {
            const res = await fetch(`/api/calcular?capacidad_van=${capacidadVan}`);
            if (!res.ok) {
                throw new Error(`Error en el servidor: ${res.statusText}`);
            }

            const data = await res.json();
            if (!data.exito) {
                throw new Error(data.error || "Error al calcular el estado.");
            }

            datosGlobales = data;
            renderTodo(data);

            setStatusBanner(
                data.modo_realtime ? "🟢 Conectado - Datos en vivo de Mugibike & Clima" : "ℹ️ Modo Histórico Offline",
                data.modo_realtime ? "success" : "info"
            );
        } catch (err) {
            console.error(err);
            setStatusBanner(`🔴 Error al calcular: ${err.message}`, "danger");
        } finally {
            setLoadingState(false);
        }
    }

    function setLoadingState(isLoading) {
        btnRecalcular.disabled = isLoading;
        if (isLoading) {
            statusText.textContent = "Calculando modelo ML y rutas en vivo...";
            statusBanner.className = "status-banner info";
        }
    }

    function setStatusBanner(msg, type) {
        statusText.textContent = msg;
        statusBanner.className = `status-banner ${type}`;
    }

    function renderTodo(data) {
        // Renderizar Tarjetas Superiores
        const r = data.resumen;
        document.getElementById('m-estaciones').textContent = r.total_estaciones;
        document.getElementById('m-bicis').textContent = r.total_bicis;
        document.getElementById('m-criticas').textContent = r.alertas_criticas;
        document.getElementById('m-clima').textContent = `${r.temperatura} °C`;
        document.getElementById('m-viento').textContent = `💨 Viento: ${r.viento_kmh} km/h`;
        document.getElementById('m-calendario').textContent = r.estado_calendario;

        // Renderizar Tabla Estaciones (Tab 1)
        renderTablaEstaciones();

        // Renderizar Ruta Furgoneta (Tab 2)
        renderRutaFurgoneta(data.ruta_furgoneta);

        // Renderizar Inactividad (Tab 3)
        renderTablaInactividad(data.inactividad);

        // Renderizar Simulación (Tab 4)
        renderTablaSimulacion(data.simulacion_impacto);

        // Renderizar Auditoría Flota (Tab 5)
        renderTablaAuditoria(data.auditoria_flota);
    }

    function renderTablaEstaciones() {
        if (!datosGlobales || !datosGlobales.estaciones) return;

        const tbody = document.getElementById('tbody-estaciones');
        const q = inputBuscar.value.toLowerCase();
        const filtro = selectFiltro.value;

        const filtradas = datosGlobales.estaciones.filter(e => {
            const coincideNombre = e.nombre_estacion.toLowerCase().includes(q);
            let coincideAlerta = true;

            if (filtro === 'critica') coincideAlerta = e.nivel_alerta.includes('CRÍTICA');
            else if (filtro === 'precaucion') coincideAlerta = e.nivel_alerta.includes('PRECAUCIÓN');
            else if (filtro === 'normal') coincideAlerta = e.nivel_alerta.includes('NORMAL');

            return coincideNombre && coincideAlerta;
        });

        if (filtradas.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center">No se encontraron estaciones.</td></tr>';
            return;
        }

        tbody.innerHTML = filtradas.map(e => {
            let badgeClass = "badge-success";
            if (e.nivel_alerta.includes('CRÍTICA')) badgeClass = "badge-danger";
            else if (e.nivel_alerta.includes('PRECAUCIÓN')) badgeClass = "badge-warning";

            return `
                <tr>
                    <td><strong>${e.nombre_estacion}</strong></td>
                    <td>${e.bicis_disponibles}</td>
                    <td>${e.capacidad}</td>
                    <td>${e.prediccion_30m} bicis</td>
                    <td><span class="badge ${badgeClass}">${e.nivel_alerta}</span></td>
                </tr>
            `;
        }).join('');
    }

    function renderRutaFurgoneta(ruta) {
        const rTiempo = document.getElementById('r-tiempo');
        const rDistancia = document.getElementById('r-distancia');
        const rParadas = document.getElementById('r-paradas');
        const tbody = document.getElementById('tbody-ruta');

        if (!ruta || !ruta.pasos || ruta.pasos.length === 0) {
            rTiempo.textContent = "0 min";
            rDistancia.textContent = "0 km";
            rParadas.textContent = "0";
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-success">🟢 Toda la red está equilibrada. No se requiere circuito de reparto en este momento.</td></tr>';
            return;
        }

        rTiempo.textContent = `${ruta.tiempo_total_min} min`;
        rDistancia.textContent = `${ruta.distancia_total_km} km`;
        rParadas.textContent = ruta.num_paradas;

        tbody.innerHTML = ruta.pasos.map((p, idx) => `
            <tr>
                <td><strong>#${idx + 1}</strong></td>
                <td>${p.estacion || p['Estación Parada'] || '--'}</td>
                <td>${p.accion || p['Acción Recomendada'] || '--'}</td>
                <td>${p.bicis_furgoneta !== undefined ? p.bicis_furgoneta : (p['Bicis en Furgoneta'] || '--')}</td>
                <td>${p.tiempo_tramo_min !== undefined ? p.tiempo_tramo_min : (p['Tiempo Tramo (min)'] || '--')} min</td>
                <td>${p.distancia_km !== undefined ? p.distancia_km : (p['Distancia (km)'] || '--')} km</td>
            </tr>
        `).join('');
    }

    function renderTablaInactividad(inactividad) {
        const tbody = document.getElementById('tbody-inactividad');
        if (!inactividad || inactividad.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center">No hay datos de inactividad disponibles.</td></tr>';
            return;
        }

        tbody.innerHTML = inactividad.map(item => `
            <tr>
                <td><strong>${item.nombre_estacion}</strong></td>
                <td>${item.horas_sin_bicis} h</td>
                <td>${item.horas_sin_anclajes} h</td>
                <td>${item.horas_inutilizada} h</td>
                <td>${item.pct_inutilizada}%</td>
                <td><span class="badge ${item.pct_inutilizada > 5 ? 'badge-warning' : 'badge-success'}">${item.tipo_indisponibilidad}</span></td>
            </tr>
        `).join('');
    }

    function renderTablaSimulacion(sim) {
        const tbody = document.getElementById('tbody-simulacion');
        if (!sim || !sim.df_estaciones_comp) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center">No hay datos de simulación disponibles.</td></tr>';
            return;
        }

        document.getElementById('sim-real').textContent = `${sim.horas_indisponible_real} h (${sim.pct_real}%)`;
        document.getElementById('sim-ml').textContent = `${sim.horas_indisponible_sim} h (${sim.pct_sim}%)`;
        document.getElementById('sim-mejora').textContent = `-${sim.mejora_pct}%`;
        document.getElementById('sim-bicis').textContent = `${sim.bicis_redistribuidas_total} bicis/mes`;

        tbody.innerHTML = sim.df_estaciones_comp.map(e => `
            <tr>
                <td><strong>${e['Estación']}</strong></td>
                <td>${e['% Inactiva (Sin Sistema)']}%</td>
                <td class="text-success"><strong>${e['% Inactiva (Con ML)']}%</strong></td>
                <td>${e['Horas Ahorradas'] || '--'} h</td>
            </tr>
        `).join('');
    }

    function renderTablaAuditoria(audit) {
        const tbody = document.getElementById('tbody-audit');
        if (!audit || !audit.historico_4am) {
            tbody.innerHTML = '<tr><td colspan="3" class="text-center">No hay datos de auditoría de flota.</td></tr>';
            return;
        }

        document.getElementById('audit-cumplimiento').textContent = `${audit.pct_cumplimiento}%`;
        document.getElementById('audit-deficit').textContent = `${audit.dias_deficit_sostenido} días`;

        tbody.innerHTML = audit.historico_4am.slice(0, 15).map(a => `
            <tr>
                <td>${a.fecha}</td>
                <td><strong>${a.bicis_operativas_4am} bicis</strong></td>
                <td>
                    <span class="badge ${a.cumple_85_pct ? 'badge-success' : 'badge-danger'}">
                        ${a.cumple_85_pct ? '✅ CUMPLE (>=43)' : '❌ DÉFICIT (<43)'}
                    </span>
                </td>
            </tr>
        `).join('');
    }

    // Carga inicial
    cargarDatos();
});
