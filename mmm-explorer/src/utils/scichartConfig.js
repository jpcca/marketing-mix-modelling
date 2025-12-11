import { SciChartSurface } from "scichart";
import { libraryVersion } from "scichart/Core/BuildStamp";

// Initialize SciChart with Community License (default behavior, but explicit check doesn't hurt)
export const initSciChart = async () => {
    // Check if a license key is available in env vars, otherwise default to community
    // if (import.meta.env.VITE_SCICHART_LICENSE) {
    //   SciChartSurface.setRuntimeLicenseKey(import.meta.env.VITE_SCICHART_LICENSE);
    // }

    SciChartSurface.UseCommunityLicense();
    // Custom theme or settings could go here
    SciChartSurface.configure({
        dataUrl: `https://cdn.jsdelivr.net/npm/scichart@${libraryVersion}/_wasm/scichart2d.data`,
        wasmUrl: `https://cdn.jsdelivr.net/npm/scichart@${libraryVersion}/_wasm/scichart2d.wasm`
    });
};

export const theme = {
    type: "Dark", // specific SciChart theme type
    // Custom colors if needed
};
