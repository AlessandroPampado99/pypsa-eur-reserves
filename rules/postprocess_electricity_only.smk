# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# Tag used to avoid colliding with sector-coupled outputs
ELECTRIC_SECTOR_OPTS_TAG = (
    config.get("electricity_only", {}).get("sector_opts_tag", "eleconly")
)


# Copy scenario dict and override sector_opts for electricity-only postprocess
ELECTRIC_SCENARIO = dict(config.get("scenario", {}))
ELECTRIC_SCENARIO["sector_opts"] = [ELECTRIC_SECTOR_OPTS_TAG]

# Ensure planning_horizons exists (some scripts/log paths expect it)
if "planning_horizons" not in ELECTRIC_SCENARIO:
    ELECTRIC_SCENARIO["planning_horizons"] = ["all"]


def electric_solved_network(w):
    """Return the solved electricity-only network produced by solve_network."""
    return RESULTS + f"networks/base_s_{w.clusters}_elec_{w.opts}.nc"


if config["foresight"] != "perfect":

    rule plot_power_network_elec_only:
        message:
            "Plotting ELECTRIC-ONLY power network for {wildcards.clusters} clusters and opts='{wildcards.opts}'"
        params:
            plotting=config_provider("plotting"),
            transmission_limit=config_provider("electricity", "transmission_limit"),
        input:
            network=electric_solved_network,
            regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        output:
            map=RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
        threads: 2
        resources:
            mem_mb=10000,
        log:
            RESULTS
            + "logs/plot_power_network_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}.log",
        benchmark:
            RESULTS
            + "benchmarks/plot_power_network_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}",
        script:
            "../scripts/plot_power_network.py"


    rule plot_balance_timeseries_elec_only:
        message:
            "Plotting ELECTRIC-ONLY balance time series for {wildcards.clusters} clusters and opts='{wildcards.opts}'"
        params:
            plotting=config_provider("plotting"),
            snapshots=config_provider("snapshots"),
            drop_leap_day=config_provider("enable", "drop_leap_day"),
        input:
            network=electric_solved_network,
            rc="matplotlibrc",
        threads: 16
        resources:
            mem_mb=10000,
        log:
            RESULTS
            + "logs/plot_balance_timeseries_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}.log",
        benchmark:
            RESULTS
            + "benchmarks/plot_balance_timeseries_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}",
        output:
            directory(
                RESULTS
                + "graphics/balance_timeseries/s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
            ),
        script:
            "../scripts/plot_balance_timeseries.py"


    rule plot_heatmap_timeseries_elec_only:
        message:
            "Plotting ELECTRIC-ONLY heatmap time series for {wildcards.clusters} clusters and opts='{wildcards.opts}'"
        params:
            plotting=config_provider("plotting"),
            snapshots=config_provider("snapshots"),
            drop_leap_day=config_provider("enable", "drop_leap_day"),
        input:
            network=electric_solved_network,
            rc="matplotlibrc",
        threads: 16
        resources:
            mem_mb=10000,
        log:
            RESULTS
            + "logs/plot_heatmap_timeseries_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}.log",
        benchmark:
            RESULTS
            + "benchmarks/plot_heatmap_timeseries_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}",
        output:
            directory(
                RESULTS
                + "graphics/heatmap_timeseries/s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
            ),
        script:
            "../scripts/plot_heatmap_timeseries.py"


    rule plot_interactive_bus_balance_elec_only:
        params:
            plotting=config_provider("plotting"),
            snapshots=config_provider("snapshots"),
            drop_leap_day=config_provider("enable", "drop_leap_day"),
            bus_name_pattern=config_provider(
                "plotting", "interactive_bus_balance", "bus_name_pattern"
            ),
        input:
            network=electric_solved_network,
            rc="matplotlibrc",
        output:
            directory=directory(
                RESULTS
                + "graphics/interactive_bus_balance/s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
            ),
        log:
            RESULTS
            + "logs/plot_interactive_bus_balance_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}.log",
        benchmark:
            RESULTS
            + "benchmarks/plot_interactive_bus_balance_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}",
        resources:
            mem_mb=20000,
        script:
            "../scripts/plot_interactive_bus_balance.py"


    rule plot_balance_map_elec_only:
        message:
            "Plotting ELECTRIC-ONLY balance map for carrier={wildcards.carrier}"
        params:
            plotting=config_provider("plotting"),
            settings=lambda w: config_provider("plotting", "balance_map", w.carrier),
        input:
            network=electric_solved_network,
            regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        output:
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}-balance_map_{carrier}.pdf",
        threads: 1
        resources:
            mem_mb=8000,
        log:
            RESULTS
            + "logs/plot_balance_map_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}_{carrier}.log",
        benchmark:
            RESULTS
            + "benchmarks/plot_balance_map_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}_{carrier}",
        script:
            "../scripts/plot_balance_map.py"


    rule plot_balance_map_interactive_elec_only:
        params:
            settings=lambda w: config_provider(
                "plotting", "balance_map_interactive", w.carrier
            ),
        input:
            network=electric_solved_network,
            regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        output:
            RESULTS
            + "maps/interactive/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}-balance_map_{carrier}.html",
        threads: 1
        resources:
            mem_mb=8000,
        log:
            RESULTS
            + "logs/plot_balance_map_interactive_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}_{carrier}.log",
        benchmark:
            RESULTS
            + "benchmarks/plot_balance_map_interactive_elec_only/"
            + "base_s_{clusters}_elec_{opts}_{sector_opts}_{planning_horizons}_{carrier}",
        conda:
            "../envs/environment.yaml",
        script:
            "../scripts/plot_balance_map_interactive.py"

ruleorder:
    plot_power_network_elec_only > plot_power_network
ruleorder:
    plot_balance_timeseries_elec_only > plot_balance_timeseries
ruleorder:
    plot_heatmap_timeseries_elec_only > plot_heatmap_timeseries
ruleorder:
    plot_interactive_bus_balance_elec_only > plot_interactive_bus_balance
ruleorder:
    plot_balance_map_elec_only > plot_balance_map
