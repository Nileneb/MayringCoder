<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class TurbulenceReport extends Model
{
    protected $keyType = 'string';
    public $incrementing = false;

    protected $fillable = [
        'id',
        'project_path',
        'summary',
        'critical_files',
        'redundancies',
    ];

    protected $casts = [
        'summary'        => 'array',
        'critical_files' => 'array',
        'redundancies'   => 'array',
    ];
}
