<?php

namespace App\Console\Commands;

use App\Models\TurbulenceReport;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Process;
use Illuminate\Support\Str;

class AnalyzeTurbulence extends Command
{
    protected $signature = 'analyze:turbulence
                            {path? : Pfad zum Repository (Standard: Projekt-Root)}
                            {--llm : LLM-Modus nutzen (langsamer, genauer; Ollama muss laufen)}
                            {--output= : JSON-Report zusätzlich als Datei speichern}';

    protected $description = 'Turbulenz-Analyse: Erkennt fragile Code-Stellen durch funktionale Farbcodierung';

    public function handle(): int
    {
        $repoPath = $this->argument('path') ?? base_path();
        $useLlm   = $this->option('llm');
        $output   = $this->option('output') ?? storage_path('app/turbulence-report.json');

        $scriptPath = base_path('scripts/analysis/turbulence_analyzer.py');

        if (! file_exists($scriptPath)) {
            $this->error("Script nicht gefunden: {$scriptPath}");
            $this->line("Bitte turbulence_analyzer.py nach scripts/analysis/ kopieren.");
            return self::FAILURE;
        }

        // Python-Befehl zusammenbauen
        $cmd = ['python3', $scriptPath, $repoPath, '--output', $output];
        if ($useLlm) {
            $cmd[] = '--llm';
        }

        $this->info("Starte Turbulenz-Analyse für: {$repoPath}");
        $this->info('Modus: ' . ($useLlm ? 'LLM (Ollama)' : 'Heuristik'));
        $this->newLine();

        // Env-Variablen aus Laravel-Config weitergeben
        $env = [
            'OLLAMA_URL'  => config('services.ollama.url', env('OLLAMA_URL', 'http://localhost:11434')),
            'TURB_MODEL'  => env('TURB_MODEL', 'mistral:7b-instruct'),
        ];

        $result = Process::env($env)
            ->timeout(600)
            ->run($cmd, function (string $type, string $buffer) {
                $this->output->write($buffer);
            });

        if (! $result->successful()) {
            $this->error('Analyse fehlgeschlagen.');
            $this->error($result->errorOutput());
            return self::FAILURE;
        }

        // JSON einlesen
        if (! file_exists($output)) {
            $this->error("Report-Datei nicht gefunden: {$output}");
            return self::FAILURE;
        }

        $json = json_decode(file_get_contents($output), true);

        if (! $json) {
            $this->error('Report-JSON konnte nicht geparst werden.');
            return self::FAILURE;
        }

        // In Datenbank speichern
        $report = TurbulenceReport::create([
            'id'             => Str::uuid(),
            'project_path'   => realpath($repoPath),
            'summary'        => $json['summary']        ?? [],
            'critical_files' => $json['critical_files'] ?? [],
            'redundancies'   => $json['redundancies']   ?? [],
        ]);

        $this->newLine();
        $this->info("Report in Datenbank gespeichert (ID: {$report->id})");
        $this->table(
            ['Metrik', 'Wert'],
            [
                ['Dateien gesamt',    $json['summary']['total_files']  ?? '-'],
                ['🔴 Kritisch (>50%)', $json['summary']['critical']     ?? '-'],
                ['🟡 Mittel (20-50%)', $json['summary']['medium']       ?? '-'],
                ['⬛ Stabil (<20%)',   $json['summary']['stable']       ?? '-'],
                ['Findings',          $json['summary']['findings']     ?? '-'],
                ['Redundanzen',       $json['summary']['redundancies'] ?? '-'],
            ]
        );

        return self::SUCCESS;
    }
}
