<?php

namespace App\Livewire;

use App\Models\TurbulenceReport as TurbulenceReportModel;
use Livewire\Component;

class TurbulenceReport extends Component
{
    public ?TurbulenceReportModel $report = null;
    public string $activeTab = 'critical';

    public function mount(?string $reportId = null): void
    {
        $this->report = $reportId
            ? TurbulenceReportModel::findOrFail($reportId)
            : TurbulenceReportModel::latest()->first();
    }

    public function loadLatest(): void
    {
        $this->report = TurbulenceReportModel::latest()->first();
    }

    public function setTab(string $tab): void
    {
        $this->activeTab = $tab;
    }

    public function render()
    {
        return view('livewire.turbulence-report', [
            'reports' => TurbulenceReportModel::latest()->limit(10)->get(['id', 'project_path', 'summary', 'created_at']),
        ]);
    }
}
