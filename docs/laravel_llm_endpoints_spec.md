# LLM-Endpoint Management — Laravel-seitige Spezifikation

**Scope:** app.linn.games soll pro Workspace eine eigene LLM-Backend-Konfiguration
verwalten (Ollama-URL, Cloud-API-Key, Modellname). MayringCoder holt die Config
per Service-Call und routet LLM-Calls entsprechend.

**MayringCoder-Seite ist fertig:** `src/llm/endpoint.py`, `src/llm/dispatch.py`,
Anthropic/OpenAI Provider. Die Laravel-Seite muss folgende Endpunkte + UI
liefern.

---

## 1. Migration `llm_endpoints`

```php
// database/migrations/2026_04_20_000001_create_llm_endpoints_table.php
<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void {
        Schema::create('llm_endpoints', function (Blueprint $t) {
            $t->id();
            $t->foreignId('workspace_id')->constrained()->cascadeOnDelete();
            $t->string('provider');            // ollama | anthropic | openai | platform
            $t->string('base_url');            // http://tailnet:11434 | https://api.anthropic.com
            $t->string('model');               // llama3.1:8b | claude-sonnet-4-6
            $t->text('api_key_encrypted')->nullable();  // Crypt::encrypt(...)
            $t->boolean('is_default')->default(false);
            $t->string('agent_scope')->nullable(); // null = alle Agents, sonst z.B. "chat-agent"
            $t->json('extra')->nullable();     // provider-spez. Zusatzfelder
            $t->timestamps();

            $t->index(['workspace_id', 'is_default']);
            $t->index(['workspace_id', 'agent_scope']);
        });
    }

    public function down(): void { Schema::dropIfExists('llm_endpoints'); }
};
```

---

## 2. Eloquent Model

```php
// app/Models/LlmEndpoint.php
<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Facades\Crypt;

class LlmEndpoint extends Model
{
    protected $fillable = [
        'workspace_id', 'provider', 'base_url', 'model',
        'api_key_encrypted', 'is_default', 'agent_scope', 'extra',
    ];

    protected $casts = [
        'is_default' => 'boolean',
        'extra'      => 'array',
    ];

    protected $hidden = ['api_key_encrypted'];

    public function workspace(): BelongsTo
    {
        return $this->belongsTo(Workspace::class);
    }

    public function apiKey(): ?string
    {
        if (!$this->api_key_encrypted) return null;
        try { return Crypt::decryptString($this->api_key_encrypted); }
        catch (\Throwable) { return null; }
    }

    public function setApiKey(?string $plain): void
    {
        $this->api_key_encrypted = $plain ? Crypt::encryptString($plain) : null;
    }

    /** Resolve the effective endpoint for a workspace + agent. */
    public static function resolveFor(Workspace $ws, ?string $agentKey = null): ?self
    {
        $q = self::query()->where('workspace_id', $ws->id);
        // 1st preference: exact agent-scoped override
        if ($agentKey) {
            $scoped = (clone $q)->where('agent_scope', $agentKey)->first();
            if ($scoped) return $scoped;
        }
        // 2nd: workspace default (agent_scope = null + is_default)
        return $q->whereNull('agent_scope')->where('is_default', true)->first();
    }
}
```

---

## 3. Service-Endpoint für MayringCoder

MayringCoder ruft auf: `GET /mcp-service/llm-endpoint/{workspace_id}?agent=<agent_key>`

Auth: `Authorization: Bearer <MCP_SERVICE_TOKEN>` (gleicher Token wie OAuth-Code-Register).

```php
// app/Http/Controllers/McpService/LlmEndpointController.php
<?php
namespace App\Http\Controllers\McpService;

use App\Http\Controllers\Controller;
use App\Models\LlmEndpoint;
use App\Models\Workspace;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;
use Symfony\Component\HttpFoundation\Response;

class LlmEndpointController extends Controller
{
    public function show(Request $request, int $workspace_id): JsonResponse
    {
        $token = $request->bearerToken() ?? '';
        $expected = (string) config('services.mcp.service_token', '');
        if ($expected === '' || !hash_equals($expected, $token)) {
            return response()->json(['error' => 'Unauthorized'], Response::HTTP_UNAUTHORIZED);
        }

        $ws = Workspace::find($workspace_id);
        if (!$ws) {
            return response()->json(['error' => 'not found'], Response::HTTP_NOT_FOUND);
        }

        $agentKey = $request->query('agent');
        $endpoint = LlmEndpoint::resolveFor($ws, $agentKey);

        if (!$endpoint) {
            // No custom config → MayringCoder falls back to platform default.
            return response()->json(['provider' => 'platform'], Response::HTTP_OK);
        }

        return response()->json([
            'provider'  => $endpoint->provider,
            'base_url'  => $endpoint->base_url,
            'model'     => $endpoint->model,
            'api_key'   => $endpoint->apiKey(),  // decrypted on-the-fly
            'extra'     => $endpoint->extra ?? (object) [],
        ]);
    }
}
```

Register in `routes/api.php`:

```php
Route::get('/mcp-service/llm-endpoint/{workspace_id}', [
    \App\Http\Controllers\McpService\LlmEndpointController::class, 'show'
])->middleware('throttle:60,1');
```

---

## 4. Livewire UI — Endpoint-Verwaltung

Route:

```php
Route::middleware(['auth', 'verified'])->group(function () {
    Route::get('einstellungen/llm-endpoints', \App\Livewire\Settings\LlmEndpoints::class)
        ->name('settings.llm-endpoints');
});
```

Livewire-Klasse:

```php
// app/Livewire/Settings/LlmEndpoints.php
<?php
namespace App\Livewire\Settings;

use App\Models\LlmEndpoint;
use Illuminate\Support\Facades\Auth;
use Livewire\Attributes\Computed;
use Livewire\Component;

class LlmEndpoints extends Component
{
    public string $provider = 'ollama';
    public string $base_url = '';
    public string $model = '';
    public string $api_key = '';
    public string $agent_scope = '';
    public bool $is_default = false;
    public ?int $editing_id = null;

    protected $rules = [
        'provider'    => 'required|in:ollama,anthropic,openai,platform',
        'base_url'    => 'required|url|max:255',
        'model'       => 'required|string|max:100',
        'api_key'     => 'nullable|string|max:500',
        'agent_scope' => 'nullable|string|max:60',
        'is_default'  => 'boolean',
    ];

    #[Computed]
    public function endpoints()
    {
        $ws = Auth::user()->currentWorkspace();
        return LlmEndpoint::where('workspace_id', $ws->id)->orderByDesc('is_default')->get();
    }

    public function save(): void
    {
        $this->validate();
        $ws = Auth::user()->currentWorkspace();

        $ep = $this->editing_id
            ? LlmEndpoint::findOrFail($this->editing_id)
            : new LlmEndpoint(['workspace_id' => $ws->id]);

        $ep->provider = $this->provider;
        $ep->base_url = rtrim($this->base_url, '/');
        $ep->model = $this->model;
        $ep->agent_scope = $this->agent_scope ?: null;
        $ep->is_default = $this->is_default;
        if ($this->api_key !== '') {
            $ep->setApiKey($this->api_key);
        }
        $ep->save();

        // Unset other defaults for the same scope
        if ($ep->is_default) {
            LlmEndpoint::where('workspace_id', $ws->id)
                ->where('agent_scope', $ep->agent_scope)
                ->where('id', '!=', $ep->id)
                ->update(['is_default' => false]);
        }

        $this->notifyMayring($ws->id);
        $this->reset(['provider', 'base_url', 'model', 'api_key', 'agent_scope', 'is_default', 'editing_id']);
        $this->dispatch('endpoint-saved');
    }

    public function edit(int $id): void
    {
        $ep = LlmEndpoint::findOrFail($id);
        $this->editing_id = $ep->id;
        $this->provider = $ep->provider;
        $this->base_url = $ep->base_url;
        $this->model = $ep->model;
        $this->agent_scope = $ep->agent_scope ?? '';
        $this->is_default = $ep->is_default;
        $this->api_key = ''; // never prefill the decrypted key
    }

    public function delete(int $id): void
    {
        $ep = LlmEndpoint::findOrFail($id);
        $wsId = $ep->workspace_id;
        $ep->delete();
        $this->notifyMayring($wsId);
    }

    /** Invalidate MayringCoder's in-process cache after any change. */
    private function notifyMayring(int $wsId): void
    {
        $mcp = rtrim((string) config('services.mayring_mcp.endpoint'), '/');
        $token = (string) config('services.mcp.service_token', '');
        if (!$mcp || !$token) return;
        try {
            \Illuminate\Support\Facades\Http::timeout(3)
                ->withToken($token)
                ->post("{$mcp}/mcp-service/llm-endpoint/{$wsId}/invalidate");
        } catch (\Throwable) { /* best-effort */ }
    }

    public function render()
    {
        return view('livewire.settings.llm-endpoints');
    }
}
```

View `resources/views/livewire/settings/llm-endpoints.blade.php`: Formular mit
Provider-Select (`ollama|anthropic|openai|platform`), URL-Input, Model-Input,
API-Key-Input (type=password, leer = unverändert), Agent-Scope-Dropdown
(`chat-agent|mayring_agent|...`), Default-Toggle. Tabelle zeigt `endpoints`-Listing
mit Edit/Delete.

---

## 5. Billing-Skip in McpAgentController

Wenn der User seinen eigenen Key mitbringt, keine Credits abbuchen.

```php
// app/Http/Controllers/McpAgentController.php — Auszug
public function call(Request $request): JsonResponse
{
    $user = $request->user();
    $workspace = $user->currentWorkspace();
    $agentKey = $request->string('agent_key');

    $endpoint = LlmEndpoint::resolveFor($workspace, $agentKey);
    $isUserManaged = $endpoint && $endpoint->provider !== 'platform';

    if (!$isUserManaged) {
        $this->credits->assertHasBalance($workspace);
        $this->credits->assertDailyLimitNotReached($workspace, $agentKey);
    }

    $response = $this->dispatchToMayring($workspace, $agentKey, $request->all());

    if (!$isUserManaged) {
        $this->credits->deduct(
            $workspace,
            (int) ($response['usage']['input_tokens'] ?? 0),
            $agentKey,
            (int) ($response['usage']['output_tokens'] ?? 0),
            (float) ($response['usage']['cost_usd'] ?? 0.0),
        );
    }

    return response()->json($response);
}
```

---

## 6. JWT ausstellen — kein Endpoint-ID nötig

MayringCoder holt Endpoint-Config per Workspace. Das bestehende RS256 JWT mit
`workspace_id` reicht. Keine zusätzliche Claim, kein Breaking Change zu
`docs/mcp_contracts.md`.

---

## 7. Env + config/services.php

```php
// config/services.php — ergänzen:
'mcp' => [
    'service_token' => env('MCP_SERVICE_TOKEN'),
    // vorhandene rate_limit/auth_token bleiben
],
```

```dotenv
# .env — gleicher Wert wie auf MayringCoder-Seite (MCP_SERVICE_TOKEN in MayringCoder .env)
MCP_SERVICE_TOKEN=<32+ Bytes Random>
```

---

## 8. Invalidate-Endpoint auf MayringCoder-Seite (optional — Phase 2)

Wenn User Config in Laravel-UI ändert, sollte der MayringCoder-Cache geflusht
werden. Das ist der `notifyMayring()`-Call oben. MayringCoder-Seite braucht
dafür eine neue Route:

```
POST /mcp-service/llm-endpoint/{workspace_id}/invalidate
Authorization: Bearer <MCP_SERVICE_TOKEN>
→ ruft src.llm.endpoint.invalidate_cache(workspace_id)
```

Diese Route ist noch nicht implementiert in MayringCoder — der 5-Minuten-TTL
fängt den Lag kurzfristig ab. Kann später nachgezogen werden.

---

## 9. Testing-Checkliste für Laravel-Seite

```bash
# Feature test: endpoint CRUD + resolveFor logic
php artisan test --filter LlmEndpointTest

# Service endpoint auth
curl -H "Authorization: Bearer $MCP_SERVICE_TOKEN" \
  https://app.linn.games/mcp-service/llm-endpoint/42
```

Pest-Test-Skelett:

```php
// tests/Feature/LlmEndpointServiceTest.php
it('returns platform provider when no config exists', function () {
    $ws = Workspace::factory()->create();
    $this->withHeader('Authorization', 'Bearer ' . config('services.mcp.service_token'))
        ->getJson("/mcp-service/llm-endpoint/{$ws->id}")
        ->assertOk()
        ->assertJson(['provider' => 'platform']);
});

it('returns encrypted-then-decrypted anthropic config', function () {
    $ws = Workspace::factory()->create();
    $ep = LlmEndpoint::create([
        'workspace_id' => $ws->id,
        'provider'     => 'anthropic',
        'base_url'     => 'https://api.anthropic.com',
        'model'        => 'claude-sonnet-4-6',
        'is_default'   => true,
    ]);
    $ep->setApiKey('sk-ant-secret');
    $ep->save();

    $this->withHeader('Authorization', 'Bearer ' . config('services.mcp.service_token'))
        ->getJson("/mcp-service/llm-endpoint/{$ws->id}")
        ->assertOk()
        ->assertJson([
            'provider' => 'anthropic',
            'api_key'  => 'sk-ant-secret',
        ]);
});
```

---

## 10. Rollout-Reihenfolge

1. Migration + Model + Unit-Tests (app.linn.games)
2. Service-Endpoint + Pest-Test
3. `MCP_SERVICE_TOKEN` in Production `.env` setzen (beide Seiten identisch)
4. Livewire-UI deployen, zunächst feature-flagged (`mayring_active + role:beta`)
5. Billing-Skip in `McpAgentController` aktivieren
6. Nach 1 Woche Observability: Flag abnehmen

**Risiko:** Wenn ein User falsche Ollama-URL konfiguriert, schlägt Generation fehl.
MayringCoder-Seitig sauber behandeln (httpx-Timeout → HTTP 502 mit Hinweistext).
Dafür existiert noch kein Guard — in Phase 2 ergänzen.
