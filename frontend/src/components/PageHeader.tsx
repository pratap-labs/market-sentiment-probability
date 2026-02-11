type PageHeaderProps = {
  title: string;
  subtitle?: string;
};

export default function PageHeader({ title, subtitle }: PageHeaderProps) {
  return (
    <div>
      <div className="page-title">{title}</div>
      {subtitle ? <div className="page-subtitle">{subtitle}</div> : null}
    </div>
  );
}
