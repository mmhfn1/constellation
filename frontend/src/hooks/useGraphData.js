// useGraphData.js
// Loads the two pieces of data needed to draw the map at all: the Tier-1
// node list (x, y, cluster_id, movie_id for every movie) and the cluster
// centroid summaries. Both are small enough to fetch once up front —
// it's rich per-movie *detail* (Tier 2) that stays lazy, fetched on
// hover/click from DetailPanel instead.
import { useEffect, useState } from "react";
import { fetchGraph, fetchClusters } from "../api.js";

export function useGraphData() {
  const [nodes, setNodes] = useState(null);
  const [clusters, setClusters] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchGraph(), fetchClusters()])
      .then(([graphRes, clusterRes]) => {
        if (cancelled) return;
        setNodes(graphRes.nodes);
        setClusters(clusterRes.clusters);
      })
      .catch((err) => {
        if (!cancelled) setError(err);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { nodes, clusters, error, isLoading: !nodes && !error };
}
