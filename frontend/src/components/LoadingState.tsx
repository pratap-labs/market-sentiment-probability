export default function LoadingState({ label = "Loading..." }: { label?: string }) {
  return (
    <div className="loading-state">
      <div className="loading-spinner" />
    </div>
  );
}
