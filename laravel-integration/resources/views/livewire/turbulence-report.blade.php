<div class="p-6 space-y-6">

    {{-- Header --}}
    <div class="flex items-center justify-between">
        <div>
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">Turbulenz-Analyse</h2>
            @if ($report)
                <p class="text-sm text-gray-500">
                    {{ $report->project_path }} &mdash;
                    {{ $report->created_at->diffForHumans() }}
                </p>
            @endif
        </div>

        {{-- Report-Auswahl --}}
        <div class="flex items-center gap-3">
            <select wire:change="mount($event.target.value)"
                    class="text-sm border-gray-300 rounded-md shadow-sm dark:bg-gray-800 dark:border-gray-600 dark:text-white">
                @foreach ($reports as $r)
                    <option value="{{ $r->id }}" @selected($report?->id === $r->id)>
                        {{ \Illuminate\Support\Str::limit($r->project_path, 40) }}
                        &mdash; {{ $r->created_at->format('d.m.Y H:i') }}
                    </option>
                @endforeach
            </select>
            <button wire:click="loadLatest"
                    class="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-md hover:bg-indigo-700">
                Aktualisieren
            </button>
        </div>
    </div>

    @if (! $report)
        <div class="rounded-md bg-yellow-50 p-4 text-sm text-yellow-800">
            Noch kein Report vorhanden. Führe
            <code class="font-mono">php artisan analyze:turbulence</code> aus.
        </div>
    @else

        {{-- Zusammenfassung --}}
        <div class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
            @php
                $s = $report->summary;
                $cards = [
                    ['label' => 'Dateien',    'value' => $s['total_files']  ?? 0, 'color' => 'gray'],
                    ['label' => '🔴 Kritisch', 'value' => $s['critical']     ?? 0, 'color' => 'red'],
                    ['label' => '🟡 Mittel',   'value' => $s['medium']       ?? 0, 'color' => 'yellow'],
                    ['label' => '⬛ Stabil',   'value' => $s['stable']       ?? 0, 'color' => 'gray'],
                    ['label' => 'Findings',   'value' => $s['findings']     ?? 0, 'color' => 'orange'],
                    ['label' => 'Redundanzen','value' => $s['redundancies'] ?? 0, 'color' => 'blue'],
                ];
            @endphp

            @foreach ($cards as $card)
                <div class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800">
                    <p class="text-xs text-gray-500 dark:text-gray-400">{{ $card['label'] }}</p>
                    <p class="mt-1 text-2xl font-bold text-gray-900 dark:text-white">{{ $card['value'] }}</p>
                </div>
            @endforeach
        </div>

        {{-- Tabs --}}
        <div class="border-b border-gray-200 dark:border-gray-700">
            <nav class="-mb-px flex gap-6" aria-label="Tabs">
                @foreach (['critical' => '🔴 Kritische Dateien', 'redundancies' => '🔄 Redundanzen'] as $tab => $label)
                    <button wire:click="setTab('{{ $tab }}')"
                            class="whitespace-nowrap border-b-2 px-1 pb-3 text-sm font-medium
                                   {{ $activeTab === $tab
                                       ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400'
                                       : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 dark:text-gray-400' }}">
                        {{ $label }}
                    </button>
                @endforeach
            </nav>
        </div>

        {{-- Kritische Dateien --}}
        @if ($activeTab === 'critical')
            @forelse ($report->critical_files as $file)
                @php
                    $pct = round($file['turbulence'] * 100);
                    $color = $pct >= 70 ? 'red' : ($pct >= 50 ? 'orange' : 'yellow');
                @endphp
                <div class="rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
                    <div class="flex items-center justify-between px-4 py-3">
                        <span class="font-mono text-sm text-gray-800 dark:text-gray-200">{{ $file['path'] }}</span>
                        <span class="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold
                                     {{ $color === 'red'    ? 'bg-red-100 text-red-800'
                                       : ($color === 'orange' ? 'bg-orange-100 text-orange-800'
                                                              : 'bg-yellow-100 text-yellow-800') }}">
                            {{ $pct }}% Turbulenz
                        </span>
                    </div>

                    {{-- Hot-Zones --}}
                    @if (! empty($file['hot_zones']))
                        <div class="border-t border-gray-100 px-4 py-2 dark:border-gray-700">
                            <p class="mb-1 text-xs font-medium text-gray-500 dark:text-gray-400">Hot-Zones</p>
                            <div class="flex flex-wrap gap-2">
                                @foreach ($file['hot_zones'] as $zone)
                                    <span class="rounded bg-red-50 px-2 py-0.5 font-mono text-xs text-red-700 dark:bg-red-900/30 dark:text-red-300">
                                        Zeile {{ $zone['start_line'] }}–{{ $zone['end_line'] }}
                                        ({{ round($zone['peak_score'] * 100) }}%)
                                    </span>
                                @endforeach
                            </div>
                        </div>
                    @endif

                    {{-- Findings --}}
                    @foreach ($file['findings'] ?? [] as $finding)
                        <div class="border-t border-gray-100 px-4 py-3 dark:border-gray-700">
                            <div class="flex items-start gap-2">
                                <span class="mt-0.5 inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium
                                             {{ ($finding['severity'] ?? '') === 'high'
                                                 ? 'bg-red-100 text-red-700'
                                                 : (($finding['severity'] ?? '') === 'medium'
                                                     ? 'bg-yellow-100 text-yellow-700'
                                                     : 'bg-gray-100 text-gray-600') }}">
                                    {{ strtoupper($finding['severity'] ?? 'low') }}
                                </span>
                                <div class="space-y-1">
                                    <p class="text-sm text-gray-800 dark:text-gray-200">
                                        <span class="font-medium">Problem:</span>
                                        {{ $finding['problem'] ?? '—' }}
                                    </p>
                                    <p class="text-sm text-gray-600 dark:text-gray-400">
                                        <span class="font-medium">Empfehlung:</span>
                                        {{ $finding['refactoring'] ?? '—' }}
                                    </p>
                                </div>
                            </div>
                        </div>
                    @endforeach
                </div>
            @empty
                <p class="text-sm text-gray-500">Keine kritischen Dateien gefunden.</p>
            @endforelse
        @endif

        {{-- Redundanzen --}}
        @if ($activeTab === 'redundancies')
            @forelse ($report->redundancies as $r)
                <div class="rounded-lg border border-blue-100 bg-blue-50/50 px-4 py-3 dark:border-blue-900/30 dark:bg-blue-900/10">
                    <div class="flex items-center justify-between">
                        <div class="space-y-0.5">
                            <p class="font-mono text-sm text-gray-800 dark:text-gray-200">
                                "{{ $r['name_a'] }}"
                                <span class="text-gray-400">in</span>
                                {{ \Illuminate\Support\Str::afterLast($r['file_a'], '/') }}
                            </p>
                            <p class="font-mono text-sm text-gray-800 dark:text-gray-200">
                                "{{ $r['name_b'] }}"
                                <span class="text-gray-400">in</span>
                                {{ \Illuminate\Support\Str::afterLast($r['file_b'], '/') }}
                            </p>
                        </div>
                        <span class="ml-4 text-lg font-bold text-blue-600 dark:text-blue-400">
                            {{ round($r['similarity'] * 100) }}%
                        </span>
                    </div>
                </div>
            @empty
                <p class="text-sm text-gray-500">Keine Redundanzen erkannt.</p>
            @endforelse
        @endif

    @endif
</div>
