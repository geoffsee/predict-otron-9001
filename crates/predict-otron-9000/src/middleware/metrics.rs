use axum::{
    extract::MatchedPath,
    http::{Request, Response},
};
use std::fmt;
use std::task::ready;
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};
use tokio::sync::Mutex;
use tower::{Layer, Service};
use tracing::{debug, info};

/// Performance metrics for a specific endpoint
#[derive(Debug, Clone, Default)]
pub struct EndpointMetrics {
    /// Total number of requests
    pub count: usize,
    /// Total response time in milliseconds
    pub total_time_ms: u64,
    /// Minimum response time in milliseconds
    pub min_time_ms: u64,
    /// Maximum response time in milliseconds
    pub max_time_ms: u64,
}

impl EndpointMetrics {
    /// Add a new response time to the metrics
    pub fn add_response_time(&mut self, time_ms: u64) {
        self.count += 1;
        self.total_time_ms += time_ms;

        if self.min_time_ms == 0 || time_ms < self.min_time_ms {
            self.min_time_ms = time_ms;
        }

        if time_ms > self.max_time_ms {
            self.max_time_ms = time_ms;
        }
    }

    /// Get the average response time in milliseconds
    pub fn avg_time_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_time_ms as f64 / self.count as f64
        }
    }

    /// Get a human-readable summary of the metrics
    pub fn summary(&self) -> String {
        format!(
            "requests: {}, avg: {:.2}ms, min: {}ms, max: {}ms",
            self.count,
            self.avg_time_ms(),
            self.min_time_ms,
            self.max_time_ms
        )
    }
}

/// Global metrics storage
#[derive(Debug, Clone, Default)]
pub struct MetricsStore {
    /// Metrics per endpoint
    endpoints: Arc<Mutex<std::collections::HashMap<String, EndpointMetrics>>>,
}

impl MetricsStore {
    /// Create a new metrics store
    pub fn new() -> Self {
        Self {
            endpoints: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Record a request's timing information
    pub async fn record(&self, path: String, time_ms: u64) {
        let mut endpoints = self.endpoints.lock().await;
        let metrics = endpoints
            .entry(path)
            .or_insert_with(EndpointMetrics::default);
        metrics.add_response_time(time_ms);
    }

    /// Get metrics for all endpoints
    pub async fn get_all(&self) -> Vec<(String, EndpointMetrics)> {
        let endpoints = self.endpoints.lock().await;
        endpoints
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Log a summary of all metrics
    pub async fn log_summary(&self) {
        let metrics = self.get_all().await;
        info!("Performance metrics summary:");

        for (path, metric) in metrics {
            info!("  {}: {}", path, metric.summary());
        }
    }
}

// Define a Layer for metrics tracking
#[derive(Debug, Clone)]
pub struct MetricsLayer {
    metrics_store: MetricsStore,
}

impl MetricsLayer {
    pub fn new(metrics_store: MetricsStore) -> Self {
        Self { metrics_store }
    }
}

impl<S> Layer<S> for MetricsLayer {
    type Service = MetricsService<S>;

    fn layer(&self, service: S) -> Self::Service {
        MetricsService {
            inner: service,
            metrics_store: self.metrics_store.clone(),
        }
    }
}

// Define a Service for metrics tracking
#[derive(Clone)]
pub struct MetricsService<S> {
    inner: S,
    metrics_store: MetricsStore,
}

impl<S> fmt::Debug for MetricsService<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetricsService")
            .field("metrics_store", &self.metrics_store)
            .finish()
    }
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for MetricsService<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
    ResBody: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        ready!(self.inner.poll_ready(cx))?;
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let path = if let Some(matched_path) = req.extensions().get::<MatchedPath>() {
            matched_path.as_str().to_string()
        } else {
            req.uri().path().to_string()
        };

        let method = req.method().clone();
        let start = Instant::now();
        let metrics_store = self.metrics_store.clone();

        let future = self.inner.call(req);

        Box::pin(async move {
            let response = future.await?;

            let time = start.elapsed();
            let status = response.status();
            let time_ms = time.as_millis() as u64;

            // Record the timing in our metrics store
            metrics_store
                .record(format!("{} {}", method, path), time_ms)
                .await;

            // Log the request timing
            debug!("{} {} {} - {} ms", method, path, status, time_ms);

            Ok(response)
        })
    }
}

/// Future that periodically logs metrics summaries
pub struct MetricsLoggerFuture {
    metrics_store: MetricsStore,
    interval: tokio::time::Interval,
}

impl MetricsLoggerFuture {
    pub fn new(metrics_store: MetricsStore, interval_secs: u64) -> Self {
        let interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));
        Self {
            metrics_store,
            interval,
        }
    }
}

impl Future for MetricsLoggerFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.interval.poll_tick(cx).is_ready() {
            let metrics_store = self.metrics_store.clone();
            tokio::spawn(async move {
                metrics_store.log_summary().await;
            });
        }

        Poll::Pending
    }
}
