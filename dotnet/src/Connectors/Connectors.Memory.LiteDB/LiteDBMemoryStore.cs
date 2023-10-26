// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using LiteDB;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Text;

namespace Microsoft.SemanticKernel.Connectors.Memory.LiteDB;

/// <summary>
/// An implementation of <see cref="IMemoryStore"/> backed by a LiteDB database.
/// </summary>
/// <remarks>The data is saved to a database file, specified in the constructor.
/// The data persists between subsequent instances. Only one instance may access the file at a time.
/// The caller is responsible for deleting the file.</remarks>
public class LiteDBMemoryStore : IMemoryStore, IDisposable
{
    /// <summary>
    /// Connect a LiteDB database
    /// </summary>
    /// <param name="filename">Path to the database file. If file does not exist, it will be created.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests. The default is <see cref="CancellationToken.None"/>.</param>
    public static Task<LiteDBMemoryStore> ConnectAsync(string filename, CancellationToken cancellationToken = default)
    {
        var memoryStore = new LiteDBMemoryStore(filename);
        return Task.FromResult(memoryStore);
    }

    /// <inheritdoc/>
    public Task CreateCollectionAsync(string collectionName, CancellationToken cancellationToken = default)
    {
        this._database.GetCollection(collectionName).Upsert(new BsonDocument());
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<bool> DoesCollectionExistAsync(string collectionName, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(this._database.CollectionExists(collectionName));
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> GetCollectionsAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.Yield();
        foreach (var collection in this._database.GetCollectionNames())
        {
            yield return collection;
        }
    }

    /// <inheritdoc/>
    public Task DeleteCollectionAsync(string collectionName, CancellationToken cancellationToken = default)
    {
        this._database.DropCollection(collectionName);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public async Task<string> UpsertAsync(string collectionName, MemoryRecord record, CancellationToken cancellationToken = default)
    {
        return await this.InternalUpsertAsync(this._database, collectionName, record, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> UpsertBatchAsync(string collectionName, IEnumerable<MemoryRecord> records,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (var record in records)
        {
            yield return await this.InternalUpsertAsync(this._database, collectionName, record, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <inheritdoc/>
    public async Task<MemoryRecord?> GetAsync(string collectionName, string key, bool withEmbedding = false, CancellationToken cancellationToken = default)
    {
        return await this.InternalGetAsync(this._database, collectionName, key, withEmbedding, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<MemoryRecord> GetBatchAsync(string collectionName, IEnumerable<string> keys, bool withEmbeddings = false,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (var key in keys)
        {
            var result = await this.InternalGetAsync(this._database, collectionName, key, withEmbeddings, cancellationToken).ConfigureAwait(false);
            if (result != null)
            {
                yield return result;
            }
            else
            {
                yield break;
            }
        }
    }

    /// <inheritdoc/>
    public async Task RemoveAsync(string collectionName, string key, CancellationToken cancellationToken = default)
    {
        await Task.Run(() => this._database.GetCollection<MemoryRecordWrapper>(collectionName)
            .DeleteMany(w => w.Key == key), cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async Task RemoveBatchAsync(string collectionName, IEnumerable<string> keys, CancellationToken cancellationToken = default)
    {
        var hashSet = new HashSet<string>(keys);
        await Task.Run(() => this._database.GetCollection<MemoryRecordWrapper>(collectionName)
            .DeleteMany(w => hashSet.Contains(w.Key)), cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<(MemoryRecord, double)> GetNearestMatchesAsync(
        string collectionName,
        ReadOnlyMemory<float> embedding,
        int limit,
        double minRelevanceScore = 0,
        bool withEmbeddings = false,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (limit <= 0)
        {
            yield break;
        }

        List<(MemoryRecord Record, double Score)> embeddings = new();

        await foreach (var record in this.GetAllAsync(collectionName, cancellationToken))
        {
            if (record != null)
            {
                double similarity = TensorPrimitives.CosineSimilarity(embedding.Span, record.Embedding.Span);
                if (similarity >= minRelevanceScore)
                {
                    var entry = withEmbeddings ? record : MemoryRecord.FromMetadata(record.Metadata, ReadOnlyMemory<float>.Empty, record.Key, record.Timestamp);
                    embeddings.Add(new(entry, similarity));
                }
            }
        }

        foreach (var item in embeddings.OrderByDescending(l => l.Score).Take(limit))
        {
            yield return (item.Record, item.Score);
        }
    }

    /// <inheritdoc/>
    public async Task<(MemoryRecord, double)?> GetNearestMatchAsync(string collectionName, ReadOnlyMemory<float> embedding, double minRelevanceScore = 0, bool withEmbedding = false,
        CancellationToken cancellationToken = default)
    {
        return await this.GetNearestMatchesAsync(
            collectionName: collectionName,
            embedding: embedding,
            limit: 1,
            minRelevanceScore: minRelevanceScore,
            withEmbeddings: withEmbedding,
            cancellationToken: cancellationToken).FirstOrDefaultAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        this.Dispose(true);
        GC.SuppressFinalize(this);
    }

    #region protected ================================================================================

    /// <summary>
    /// Disposes the resources used by the <see cref="LiteDBMemoryStore"/> instance.
    /// </summary>
    /// <param name="disposing">True to release both managed and unmanaged resources; false to release only unmanaged resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!this._disposedValue)
        {
            if (disposing)
            {
                this._database.Dispose();
            }

            this._disposedValue = true;
        }
    }

    #endregion

    #region private ================================================================================

    private readonly LiteDatabase _database;
    private bool _disposedValue;

    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="filename">LiteDB db filename.</param>
    private LiteDBMemoryStore(string filename)
    {
        BsonMapper.Global.RegisterType<ReadOnlyMemory<float>>(
            readOnlyMemory => new BsonValue(System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(readOnlyMemory, CreateSerializerOptions())),
            bson => System.Text.Json.JsonSerializer.Deserialize<ReadOnlyMemory<float>>(bson.AsBinary, CreateSerializerOptions()));
        this._database = new LiteDatabase(filename, BsonMapper.Global);
        this._disposedValue = false;
    }

    private async IAsyncEnumerable<MemoryRecord> GetAllAsync(string collectionName, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.Yield();
        foreach (var record in this._database.GetCollection<MemoryRecordWrapper>(collectionName).FindAll())
        {
            if (string.IsNullOrEmpty(record.Key) || string.IsNullOrEmpty(record.Metadata))
            {
                continue;
            }

            yield return record.ToMemoryRecord();
        }
    }

    private Task<string> InternalUpsertAsync(LiteDatabase connection, string collectionName, MemoryRecord record, CancellationToken cancellationToken)
    {
        record.Key = record.Metadata.Id;
        connection.GetCollection<MemoryRecordWrapper>(collectionName).Upsert(record.Key, MemoryRecordWrapper.FromMemoryRecord(record));
        return Task.FromResult(record.Key);
    }

    private Task<MemoryRecord?> InternalGetAsync(
        LiteDatabase connection,
        string collectionName,
        string key, bool withEmbedding,
        CancellationToken cancellationToken)
    {
        var record = connection.GetCollection<MemoryRecordWrapper>(collectionName).FindOne(f => f.Key == key);
        return Task.FromResult(record?.ToMemoryRecord(withEmbedding));
    }

    private static JsonSerializerOptions CreateSerializerOptions()
    {
        var jso = new JsonSerializerOptions();
        jso.Converters.Add(new ReadOnlyMemoryConverter());
        return jso;
    }

    private sealed class MemoryRecordWrapper
    {
        public string Key = string.Empty;
        public string Metadata = string.Empty;
        public ReadOnlyMemory<float> Embedding;
        public DateTimeOffset? Timestamp;

        public MemoryRecord ToMemoryRecord(bool withEmbedding = true)
        {
            return MemoryRecord.FromJsonMetadata(this.Metadata, withEmbedding ? this.Embedding : ReadOnlyMemory<float>.Empty, this.Key, this.Timestamp);
        }

        public static MemoryRecordWrapper FromMemoryRecord(MemoryRecord record)
        {
            return new()
            {
                Key = record.Key,
                Metadata = record.GetSerializedMetadata(),
                Embedding = record.Embedding,
                Timestamp = record.Timestamp
            };
        }
    }

    #endregion
}
