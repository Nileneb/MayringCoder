<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('turbulence_reports', function (Blueprint $table) {
            $table->uuid('id')->primary();
            $table->string('project_path');
            $table->json('summary');
            $table->json('critical_files');
            $table->json('redundancies');
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('turbulence_reports');
    }
};
