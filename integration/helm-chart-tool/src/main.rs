use anyhow::{Context, Result};
use clap::{Arg, Command};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Debug, Deserialize)]
struct CargoToml {
    package: Option<Package>,
}

#[derive(Debug, Deserialize)]
struct Package {
    name: String,
    metadata: Option<Metadata>,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    kube: Option<KubeMetadata>,
}

#[derive(Debug, Deserialize)]
struct KubeMetadata {
    image: String,
    replicas: Option<u32>,
    port: u16,
}


#[derive(Debug, Clone)]
struct ServiceInfo {
    name: String,
    image: String,
    port: u16,
    replicas: u32,
}

fn main() -> Result<()> {
    let matches = Command::new("helm-chart-tool")
        .about("Generate Helm charts from Cargo.toml metadata")
        .arg(
            Arg::new("workspace")
                .short('w')
                .long("workspace")
                .value_name("PATH")
                .help("Path to the workspace root")
                .default_value("."),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("PATH")
                .help("Output directory for the Helm chart")
                .default_value("./helm-chart"),
        )
        .arg(
            Arg::new("chart-name")
                .short('n')
                .long("name")
                .value_name("NAME")
                .help("Name of the Helm chart")
                .default_value("predict-otron-9000"),
        )
        .get_matches();

    let workspace_path = matches.get_one::<String>("workspace").unwrap();
    let output_path = matches.get_one::<String>("output").unwrap();
    let chart_name = matches.get_one::<String>("chart-name").unwrap();

    println!("Parsing workspace at: {}", workspace_path);
    println!("Output directory: {}", output_path);
    println!("Chart name: {}", chart_name);

    let services = discover_services(workspace_path)?;
    println!("Found {} services:", services.len());
    for service in &services {
        println!(
            "  - {}: {} (port {})",
            service.name, service.image, service.port
        );
    }

    generate_helm_chart(output_path, chart_name, &services)?;
    println!("Helm chart generated successfully!");

    Ok(())
}

fn discover_services(workspace_path: &str) -> Result<Vec<ServiceInfo>> {
    let workspace_root = Path::new(workspace_path);
    let mut services = Vec::new();

    // Find all Cargo.toml files in the workspace
    for entry in WalkDir::new(workspace_root)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_name() == "Cargo.toml"
            && entry.path() != workspace_root.join("../../../Cargo.toml")
        {
            if let Ok(service_info) = parse_cargo_toml(entry.path()) {
                services.push(service_info);
            }
        }
    }

    Ok(services)
}

fn parse_cargo_toml(path: &Path) -> Result<ServiceInfo> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read Cargo.toml at {:?}", path))?;

    let cargo_toml: CargoToml = toml::from_str(&content)
        .with_context(|| format!("Failed to parse Cargo.toml at {:?}", path))?;

    let package = cargo_toml
        .package
        .ok_or_else(|| anyhow::anyhow!("No package section found in {:?}", path))?;

    let metadata = package
        .metadata
        .ok_or_else(|| anyhow::anyhow!("No metadata section found in {:?}", path))?;

    let kube_metadata = metadata
        .kube
        .ok_or_else(|| anyhow::anyhow!("No kube metadata found in {:?}", path))?;

    Ok(ServiceInfo {
        name: package.name,
        image: kube_metadata.image,
        port: kube_metadata.port,
        replicas: kube_metadata.replicas.unwrap_or(1),
    })
}

fn generate_helm_chart(
    output_path: &str,
    chart_name: &str,
    services: &[ServiceInfo],
) -> Result<()> {
    let chart_dir = Path::new(output_path);
    let templates_dir = chart_dir.join("templates");

    // Create directories
    fs::create_dir_all(&templates_dir)?;

    // Generate Chart.yaml
    generate_chart_yaml(chart_dir, chart_name)?;

    // Generate values.yaml
    generate_values_yaml(chart_dir, services)?;

    // Generate templates for each service
    for service in services {
        generate_deployment_template(&templates_dir, service)?;
        generate_service_template(&templates_dir, service)?;
    }

    // Generate ingress template
    generate_ingress_template(&templates_dir, services)?;

    // Generate helper templates
    generate_helpers_template(&templates_dir)?;

    // Generate .helmignore
    generate_helmignore(chart_dir)?;

    Ok(())
}

fn generate_chart_yaml(chart_dir: &Path, chart_name: &str) -> Result<()> {
    let chart_yaml = format!(
        r#"apiVersion: v2
name: {}
description: A Helm chart for the predict-otron-9000 AI platform
type: application
version: 0.1.0
appVersion: "0.1.0"
keywords:
  - ai
  - llm
  - inference
  - embeddings
  - chat
maintainers:
  - name: predict-otron-9000-team
"#,
        chart_name
    );

    fs::write(chart_dir.join("Chart.yaml"), chart_yaml)?;
    Ok(())
}

fn generate_values_yaml(chart_dir: &Path, services: &[ServiceInfo]) -> Result<()> {
    let mut values = String::from(
        r#"# Default values for predict-otron-9000
# This is a YAML-formatted file.

global:
  imagePullPolicy: IfNotPresent
  serviceType: ClusterIP

# Ingress configuration
ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: predict-otron-9000.local
      paths:
        - path: /
          pathType: Prefix
          backend:
            service:
              name: predict-otron-9000
              port:
                number: 8080
  tls: []

"#,
    );

    for service in services {
        let service_config = format!(
            r#"{}:
  image:
    repository: {}
    tag: "latest"
    pullPolicy: IfNotPresent
  replicas: {}
  service:
    type: ClusterIP
    port: {}
  resources:
    limits:
      memory: "1Gi"
      cpu: "1000m"
    requests:
      memory: "512Mi"
      cpu: "250m"
  nodeSelector: {{}}
  tolerations: []
  affinity: {{}}

"#,
            service.name.replace("-", "_"),
            service.image.split(':').next().unwrap_or(&service.image),
            service.replicas,
            service.port
        );
        values.push_str(&service_config);
    }

    fs::write(chart_dir.join("values.yaml"), values)?;
    Ok(())
}

fn generate_deployment_template(templates_dir: &Path, service: &ServiceInfo) -> Result<()> {
    let service_name_underscore = service.name.replace("-", "_");
    let deployment_template = format!(
        r#"apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{{{ include "predict-otron-9000.fullname" . }}}}-{}
  labels:
    {{{{- include "predict-otron-9000.labels" . | nindent 4 }}}}
    app.kubernetes.io/component: {}
spec:
  replicas: {{{{ .Values.{}.replicas }}}}
  selector:
    matchLabels:
      {{{{- include "predict-otron-9000.selectorLabels" . | nindent 6 }}}}
      app.kubernetes.io/component: {}
  template:
    metadata:
      labels:
        {{{{- include "predict-otron-9000.selectorLabels" . | nindent 8 }}}}
        app.kubernetes.io/component: {}
    spec:
      containers:
        - name: {}
          image: "{{{{ .Values.{}.image.repository }}}}:{{{{ .Values.{}.image.tag }}}}"
          imagePullPolicy: {{{{ .Values.{}.image.pullPolicy }}}}
          ports:
            - name: http
              containerPort: {}
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{{{- toYaml .Values.{}.resources | nindent 12 }}}}
      {{{{- with .Values.{}.nodeSelector }}}}
      nodeSelector:
        {{{{- toYaml . | nindent 8 }}}}
      {{{{- end }}}}
      {{{{- with .Values.{}.affinity }}}}
      affinity:
        {{{{- toYaml . | nindent 8 }}}}
      {{{{- end }}}}
      {{{{- with .Values.{}.tolerations }}}}
      tolerations:
        {{{{- toYaml . | nindent 8 }}}}
      {{{{- end }}}}
"#,
        service.name,
        service.name,
        service_name_underscore,
        service.name,
        service.name,
        service.name,
        service_name_underscore,
        service_name_underscore,
        service_name_underscore,
        service.port,
        service_name_underscore,
        service_name_underscore,
        service_name_underscore,
        service_name_underscore
    );

    let filename = format!("{}-deployment.yaml", service.name);
    fs::write(templates_dir.join(filename), deployment_template)?;
    Ok(())
}

fn generate_service_template(templates_dir: &Path, service: &ServiceInfo) -> Result<()> {
    let service_template = format!(
        r#"apiVersion: v1
kind: Service
metadata:
  name: {{{{ include "predict-otron-9000.fullname" . }}}}-{}
  labels:
    {{{{- include "predict-otron-9000.labels" . | nindent 4 }}}}
    app.kubernetes.io/component: {}
spec:
  type: {{{{ .Values.{}.service.type }}}}
  ports:
    - port: {{{{ .Values.{}.service.port }}}}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{{{- include "predict-otron-9000.selectorLabels" . | nindent 4 }}}}
    app.kubernetes.io/component: {}
"#,
        service.name,
        service.name,
        service.name.replace("-", "_"),
        service.name.replace("-", "_"),
        service.name
    );

    let filename = format!("{}-service.yaml", service.name);
    fs::write(templates_dir.join(filename), service_template)?;
    Ok(())
}

fn generate_ingress_template(templates_dir: &Path, _services: &[ServiceInfo]) -> Result<()> {
    let ingress_template = r#"{{- if .Values.ingress.enabled -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "predict-otron-9000.fullname" . }}
  labels:
    {{- include "predict-otron-9000.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            {{- if .pathType }}
            pathType: {{ .pathType }}
            {{- end }}
            backend:
              service:
                name: {{ include "predict-otron-9000.fullname" $ }}-{{ .backend.service.name }}
                port:
                  number: {{ .backend.service.port.number }}
          {{- end }}
    {{- end }}
{{- end }}
"#;

    fs::write(templates_dir.join("ingress.yaml"), ingress_template)?;
    Ok(())
}

fn generate_helpers_template(templates_dir: &Path) -> Result<()> {
    let helpers_template = r#"{{/*
Expand the name of the chart.
*/}}
{{- define "predict-otron-9000.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "predict-otron-9000.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "predict-otron-9000.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "predict-otron-9000.labels" -}}
helm.sh/chart: {{ include "predict-otron-9000.chart" . }}
{{ include "predict-otron-9000.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "predict-otron-9000.selectorLabels" -}}
app.kubernetes.io/name: {{ include "predict-otron-9000.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "predict-otron-9000.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "predict-otron-9000.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
"#;

    fs::write(templates_dir.join("_helpers.tpl"), helpers_template)?;
    Ok(())
}

fn generate_helmignore(chart_dir: &Path) -> Result<()> {
    let helmignore_content = r#"# Patterns to ignore when building packages.
# This supports shell glob matching, relative path matching, and
# negation (prefixed with !). Only one pattern per line.
.DS_Store
# Common VCS dirs
.git/
.gitignore
.bzr/
.bzrignore
.hg/
.hgignore
.svn/
# Common backup files
*.swp
*.bak
*.tmp
*.orig
*~
# Various IDEs
.project
.idea/
*.tmproj
.vscode/
"#;

    fs::write(chart_dir.join(".helmignore"), helmignore_content)?;
    Ok(())
}
